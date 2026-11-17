# L1 Ledger: genify_matmul

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes, from the Blocking Model in `op_design.md`:
- **b**: `block_batches`
- **m**: `block_m_tiles`
- **n**: `block_n_tiles`; the output width is `out_block_n_tiles = block_n_tiles / 2` for swiglu, else `block_n_tiles`
- **k**: `block_k_tiles`, with `num_k_blocks` blocks

Phase 0 (regime R1, epilogue none) allocates the first four rows. The later rows are listed so that the extent solve and the sharing decisions are already priced.

## CB table

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_act_tiles` | `depth_act · block_m_tiles · block_k_tiles` | `block_m_tiles · block_k_tiles` (one K-block in compute) + one in flight from the reader | `{b: streams, m: spans → block_m_tiles, n: streams (re-read per N-block), k: spans → block_k_tiles (window of one K-block)}` | A dtype (bf16 / bf8b / fp32 since Refinement 1) | reader | compute | whole kernel | **None.** `depth_act = 2` is double buffering (the read of K-block kb+1 overlaps the multiply of kb). It cannot alias `cb_weight_tiles`: both are live concurrently, as in0 and in1 of the same matmul (the helper requires distinct CBs, `matmul_block_helpers.hpp:242-246`). |
| `cb_weight_tiles` | `depth_weight · block_k_tiles · block_n_tiles` | `block_k_tiles · block_n_tiles` + one in flight | `{b: streams, m: streams (re-read per M-block), n: spans → block_n_tiles, k: spans → block_k_tiles}` | B dtype (bf16 / bf8b / bf4b since Refinement 1) | reader | compute | whole kernel | **None.** `depth_weight = 2` is double buffering. Same concurrency as `cb_act_tiles`. |
| `cb_partials` | `block_m_tiles · block_n_tiles` (exact, a mechanism cap: in-place packer L1 accumulation needs the ring to return to the block base). Allocated **only if** `num_k_blocks > 1 or epilogue != none`. | the whole block: every output tile's running sum is live across all K-blocks | `{b: streams (per batch slice), m: spans → block_m_tiles, n: spans → block_n_tiles, k: streams (accumulated in place)}` | `Float32` when `fp32_dest_acc_en` (the K-accumulator must keep fp32 DEST precision across K-blocks), else `Float16_b` | compute | compute | one block | **None.** It cannot share with `cb_output_tiles`: the page formats differ (Float32 vs the output dtype), and on the last K-block both are live (the partials are reloaded while output row groups are packed). It cannot fold into DEST: the block (up to ~200 tiles) exceeds DEST (4 tiles), which is the reason the accumulator exists. When `num_k_blocks == 1` and epilogue none, the CB is omitted entirely and `cb_output_tiles` is the helper's placeholder. |
| `cb_output_tiles` | `depth_out · sb_h · out_block_n_tiles` | one row group `sb_h · out_block_n_tiles` being packed + one draining | `{b: streams, m: streams (row groups of sb_h), n: spans → out_block_n_tiles, k: streams (written once, after the last K-block)}` | output dtype (the `dtype` override, else A dtype) | compute | writer | whole kernel | **None.** `depth_out = 2` is double buffering against the writer. Packing straight into this destination is already the reuse (pattern 1). It cannot alias `cb_partials` (different format and concurrent use, see above). |
| `cb_bias_row` (bias, bias_relu; built in Refinement 2) | `block_n_tiles` | `block_n_tiles` (the whole N-block row, read for every row group) | `{b: streams, m: streams (broadcast over m), n: spans → block_n_tiles, k: streams (unused)}` | bias dtype | reader | compute | one block | **None.** It is live across the whole epilogue while `cb_partials` and `cb_output_tiles` are both in use. Single-buffered: one fetch per block, so a second slot buys almost nothing (a lamp if the per-block bias read ever shows up in a profile). |
| `cb_gate_row` (addcmul; built in Refinement 2) | `block_n_tiles` | `block_n_tiles` | `{b: streams, m: streams (broadcast), n: spans → block_n_tiles, k: streams}` | gate dtype (= output dtype) | reader | compute | one block | **None.** Concurrent with `cb_partials` and `cb_residual_tiles` in the same chain step. Only one of `cb_bias_row` / `cb_gate_row` exists per program (the epilogues are exclusive), so they **alias by construction**: the same index range is never allocated twice. |
| `cb_residual_tiles` (addcmul; built in Refinement 2) | `depth_residual · sb_h · block_n_tiles` | one row group + one in flight | `{b: streams, m: streams (row groups), n: spans → block_n_tiles, k: streams}` | output dtype | reader | compute | whole kernel | **None.** It streams concurrently with `cb_output_tiles`, so the reader and the writer would both touch one CB, which breaks the ownership rule. `depth_residual = 2` is double buffering. |
| `cb_act_shard` (later: R2, R3) | the A shard's tiles (`cb_descriptor_from_sharded_tensor`) | the whole shard (resident) | `{b: streams, m: spans → shard_m_tiles (= block_m_tiles), n: streams, k: spans → shard_k_tiles}` | A dtype | reader (one `cb_push_back`, no NoC) | compute | kernel | **Backed on the A tensor's own L1**, so it adds 0 bytes to the CB budget. It replaces `cb_act_tiles` in R2 (that CB is then not allocated). R3 keeps `cb_act_tiles` for the peer K-slices; same format, so `In0SourceFn` can select either. |
| `cb_output_shard` (later: R2, R3, R4 when `block_n_tiles == out shard width`) | the output shard's tiles | the whole shard | `{b: streams, m: spans → block_m_tiles, n: spans → block_n_tiles, k: streams}` | output dtype | compute | writer (wait/pop only) | kernel | **Backed on the output tensor's L1**, 0 extra bytes. It replaces `cb_output_tiles`. |

**Refinement 2 (epilogues) as built.**
- `cb_partials` is allocated whenever epilogue ≠ none, still exactly one block. The matmul packs its last K-block there (`LastBlockTarget::Interm`), and the epilogue drains it into `cb_output_tiles`.
- `_footprint()` now carries every epilogue term below (`E_bias` / `E_addcmul`, and `out_block_n = block_n/2` for swiglu), so the S1–S3 solve prices them. `_block_k_max()` derives its fixed part from `_footprint(…, bk=0)`, which keeps the formula in one place.
- `cb_bias_row` / `cb_gate_row` stay single-buffered. The reader pushes the row **after the first batch's K-walk of the block**, not before it. With one slot, a push ahead of the K-blocks would park the reader on `cb_reserve_back` until the previous block's epilogue popped the row, which would stall the next block's K-block prefetch for a whole epilogue.
- `cb_residual_tiles` row groups are pushed after each batch's K-walk, in epilogue order.
- Ablation, PERF FOCUS 1: the Interm/partials plumbing adds no measurable time. With the chain compute stubbed the run takes 1564 µs, against 1572 µs for the epilogue-none twin. The fused swiglu costs only its SFPU compute.

## Symbol table (non-block parameters and their bounds)

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `depth_act`, `depth_weight`, `depth_out`, `depth_residual` | = 2 (Phase 0 constants, knobs) | host constants |
| `sb_h` | `1 ≤ sb_h ≤ dest_limit`, `sb_h ∣ block_m_tiles` | host subblock derivation (`dest_limit = 4` fp32 DEST / 8) |
| `a_tile_bytes`, `b_tile_bytes`, `o_tile_bytes`, `bias_tile_bytes` | ≤ 4096 (fp32 tile) | `tensor.buffer_page_size()` / `ttnn.tile_size(dtype)` |
| `p_tile_bytes` | 4096 (fp32 DEST) or 2048 | `fp32_dest_acc_en` |
| `block_m_tiles` | ≤ `core_m_tiles` = `ceil(tensor_m_tiles / grid_m)` | the L1 solve (S1–S3) keeps `F ≤ l1_cb_budget` |
| `block_n_tiles` | ≤ `core_n_tiles` (multiple of `q_n`) | the L1 solve |
| `block_k_tiles` | `min(tensor_k_tiles, 8, block_k_cap) ≤ block_k_tiles ≤ block_k_cap` | the L1 solve (S2 floor `block_k_floor`), capped by `block_k_cap` |
| `block_k_cap` | `tensor_k_tiles` with fp32 DEST; `min(tensor_k_tiles, DEST16_BLOCK_K_MAX = 8)` with 16-bit DEST | Refinement 1 precision knob: a bf16 DEST running sum swamps small addends, so the in-DEST K chain is capped and the cross-K-block sum goes to the packer (`cb_partials`, Float16_b). With 16-bit DEST and K > 8 tiles, `cb_partials` therefore always exists (S1 "one resident K-block" applies only when `tensor_k_tiles ≤ block_k_cap`). |
| `tensor_k_tiles`, `tensor_m_tiles`, `tensor_n_tiles` | **not** in any capacity expression | — (the op dimensions enter only through the bounded block extents) |
| `l1_cb_budget` | `min(lowest allocated L1 block offset in ttnn.get_memory_view(device, L1).block_table (bank size if none), total_bytes_per_bank − l1_tensor_bytes_per_core) − l1_reserve_bytes`. Verifier fix: the earlier `total − allocated` ignored fragmentation (a freed tensor above a live one) and let CBs clash with L1 buffers (implementer: `get_max_worker_l1_unreserved_size()` counts from the kernel-config ring buffer, below the CB base `l1_unreserved_base`, and over-stated the budget by ~70 KB on BH → CB overflow on the large loose cases; 1460288 B on p300c) | host `l1_cb_budget()`; `l1_reserve_bytes = 0` in Phase 0 |

## Total per-core footprint (closed form)

```
F(block_m, block_n, block_k) =
      depth_act    · block_m · block_k · a_tile_bytes                       # scales with m·k
    + depth_weight · block_k · block_n · b_tile_bytes                       # scales with k·n
    + P            · block_m · block_n · p_tile_bytes                       # scales with m·n;  P = [num_k_blocks>1 or epilogue!=none]
    + depth_out    · sb_h · out_block_n · o_tile_bytes                      # scales with n
    + E_bias       · block_n · bias_tile_bytes                              # scales with n  (bias epilogues)
    + E_addcmul    · (block_n · o_tile_bytes + depth_residual · sb_h · block_n · o_tile_bytes)   # scales with n
  ≤ l1_cb_budget
```

`F` is affine in `block_k` for fixed (m, n). So S2's `block_k_tiles_max` is a closed-form division:
`(budget − terms without k) / (depth_act·block_m·a + depth_weight·block_n·b)`.

Worked values, 11×10 grid, budget ≈ 1.4 MB (illustrative; the host computes the real number):

| Shape (A × B, dtypes) | Region / block | Terms | `block_k` |
|---|---|---|---|
| prefill gate_up 2048×5120 @ 5120×8704 (bf16 × bf4b, swiglu, regime R1 → R1a later) | grid 11×10, block 6×28 | partials 168 × 4 KB = 672 KB; out 2·1·14·2 KB = 56 KB; per `bk` 2·(6·2 KB + 28·576 B) = 56.8 KB | `num_k_blocks = 15`, `block_k = 11`, `last = 6` |
| 1×6144 @ 6144×6144 (bf16, 1-D grid) | `grid_n' = 96`, block 1×2 | partials 8 KB; out 8 KB; per `bk` 2·(2 KB + 4 KB) | `num_k_blocks = 2`, `block_k = 96` |
| (1,1,2048,256) @ 256×512 (bf16) | grid 11×8, block 6×2 | S1 fits: A 2·6·8·2 KB + B 2·8·2·2 KB + out 2·2·2 KB ≈ 264 KB | `num_k_blocks = 1`, no `cb_partials` |

## Data-movement budget (chosen split R1, interleaved)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| A | `grid_n' · nb_n` | Each core reads its own `block_m` rows per N-block; nothing is shared between the cores of a grid row (stepping stone) | none |
| B | `grid_m' · nb_m` | Each core reads its own `block_n` columns per M-block | none |
| output | 1 | Written once, after the last K-block (the K-sum stays in `cb_partials`, so there is no read-modify-write in DRAM) | none |
| bias / gate | `grid_m' · nb_m` (one row per block, ≤ `block_n` tiles) | reuse-shared row, re-read per block | none |
| residual | 1 | streamed once alongside the output | none |

**Total bytes per tier, prefill gate_up example** (A bf16 21.0 MB, B bf4b 25.1 MB, out bf16 17.8 MB, grid 11×10 with `grid_m' = 11`, `grid_n' = 10`, `nb = 1`):
- DRAM: A ×10 = 210 MB, B ×11 = 276 MB, out 17.8 MB, total ≈ 504 MB
- cross-core: 0
- core-local: each K-block is read once from its CB; the partials are read and written once per K-block per tile (L1 accumulation)

**1-D (decode-shaped) example** (1×6144 @ 6144×6144 bf16, `grid_n' = 96`):
- DRAM: A ×96 = 37.7 MB, B ×1 = 75.5 MB, out 12 KB, total ≈ 113 MB

> Cheapest-traffic split considered: `R1a interleaved_2d_mcast` (and `R1b interleaved_stream_mcast_a` for 1-D grids). Delta for prefill: DRAM 504 MB → 64 MB (−440 MB), adding NoC multicasts of each A K-block to `grid_n'−1` receivers per row and each B K-block to `grid_m'−1` per column. Delta for the 1-D example: DRAM 113 MB → 75.5 MB (−37.7 MB), adding a broadcast of A (≤ 0.4 MB) to 96 cores.
> Implemented: `R1 interleaved_out_split`.
> Deferred because R1 is the stepping stone both successors are built on. They keep R1's per-core region, CB inventory (no new CB: the multicast lands in `cb_act_tiles` / `cb_weight_tiles`), compute kernel and block walk; only the reader's source for one operand changes from DRAM to an injector. R1's factorization already maps the logical `(grid_m', grid_n')` onto a physical rectangle, so the multicast groups (grid rows and columns) exist unchanged.
