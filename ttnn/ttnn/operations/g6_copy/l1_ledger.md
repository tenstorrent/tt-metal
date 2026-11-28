# L1 Ledger: g6_copy

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes from `op_design.md` →
Blocking Model: `page` (the block axis), `batch` / `tile_row` / `tile_col` (folded into `page`),
`intra_page` (whole page, not split).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_page_staging` | `cb_depth_blocks · block_pages` (since Refinement 2 no longer the whole assignment on large shapes: `depth · block_pages` with `block_pages = BLOCK_BYTES // page_bytes`, 2 × 4 fp32 pages = 32,768 B on (16, 12, 128, 128)). `one_block` plan (assignment ≤ one `BLOCK_BYTES` block): `block_pages = core_pages_max`, `cb_depth_blocks = 1`, guarded by `core_pages_max · page_bytes ≤ L1_BUDGET_BYTES`. `streamed` plan: `block_pages = max(1, min(block_target or ∞, L1_BUDGET_BYTES // (STREAM_DEPTH_BLOCKS · page_bytes)))`, `cb_depth_blocks = min(STREAM_DEPTH_BLOCKS, L1_BUDGET_BYTES // (block_pages · page_bytes))` | **= capacity.** `one_block`: the whole block is live from the last read landing until the writes flush. `streamed`: block k (being written) and block k+1 (being read) are live at once. No gap to account for | `{page: spans → block_pages × cb_depth_blocks; batch: folded into page (spanned through page); tile_row: folded into page (spanned through page); tile_col: folded into page (spanned through page); intra_page: spans → whole page (page_bytes per slot)}` | the tensor's storage format, never interpreted: float32 / bfloat16 / bfloat8_b / int32 / uint32 tiles (4096 / 2048 / 1088 / 4096 / 4096 B) or ROW_MAJOR rows of bfloat16 / int32 / uint32 (`round_up(W · elem_bytes, 64)` B; a 4 B row is a 64 B page) — all supported since Refinement 1. Audit 2 does not apply: a dataflow-only payload that never passes through DEST, in an op with no `fp32_dest_acc_en`. The bytes must stay bit-identical, so any other width would violate the contract | reader | writer | whole kernel (`load_block` → `store_block`, every block) | **The only CB, so there is nothing to share with, and it cannot be removed:** a Tensix NoC read lands only in the issuing core's L1 (dataflow_api.h:526-530), so DRAM → DRAM needs an L1 hop. Pattern 1 (pack into destination): n/a, the destination is DRAM. Pattern 2 (in place): this is the single buffer. Pattern 3 (alias): no second buffer. Pattern 4 (fold into DEST): no compute |

## Symbol table

| Symbol | Meaning | Bound | Predicate establishing the bound |
|--------|---------|-------|----------------------------------|
| `page_bytes` | `buffer_aligned_page_size()` | TILE: 1088 (bfp8), 2048 (bf16), 4096 (fp32 / int32 / uint32). ROW_MAJOR: `round_up(W · elem_bytes, 64)`. Always ≤ `L1_BUDGET_BYTES` | host refusal when `page_bytes > L1_BUDGET_BYTES` (BB-05); unreachable for TILE; `sub_page_rows` is the deferred regime past it |
| `tensor_pages` | `buffer_num_pages()` | unbounded op dimension | never enters a capacity directly, only through `core_pages_max` |
| `num_cores` | `min(tensor_pages, pool_cores)` | ≤ grid cores (110 here) | `split_work_to_cores` over the pool; `pool_cores` = grid or `MAX_CORES` |
| `core_pages_max` | `ceil(tensor_pages / num_cores)` | unbounded | enters capacity only under the `one_block` predicate `core_pages_max · page_bytes ≤ L1_BUDGET_BYTES` |
| `L1_BUDGET_BYTES` | L1 per core left to the op by the model | 1347776 (constant) | model environment / BB-05 |
| `BLOCK_BYTES` | bytes per block (overlap knob); `block_target = BLOCK_BYTES // page_bytes` | 16384 (Refinement 2) | knob |
| `MAX_BLOCK_PAGES` | block-extent cap (forced tests); `block_target = min(…, MAX_BLOCK_PAGES)` | `None` (no cap) | knob |
| `STREAM_DEPTH_BLOCKS` | depth on the `streamed` plan | 2 (constant) | knob |

**Total per-core footprint:** `L1_op = cb_depth_blocks · block_pages · page_bytes ≤ L1_BUDGET_BYTES`.
- Scales with the `page` block knob (`block_pages`) and the depth knob only. Nothing else is
  allocated: no other CB, no semaphores, no L1 buffers.
- `one_block`: `L1_op = core_pages_max · page_bytes ≤ L1_BUDGET_BYTES` by its predicate.
  `streamed`: `L1_op ≤ min(STREAM_DEPTH_BLOCKS · block_pages, L1_BUDGET_BYTES // page_bytes) · page_bytes ≤ L1_BUDGET_BYTES` by the clamp.
- Since Refinement 2 the largest per-core footprint is 32,768 B: `streamed` at 2 × 16 KiB (float32
  (16|8, 12, 128, 128), bfloat16 (1, 1, 32, 62080) / (1, 1, 8448, 128)); everything else is a
  `one_block` ≤ 16 KiB.

## Data-movement budget

Chosen split: `page` in flat contiguous page ranges over the full grid, with coalesced bank-run
transfers.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input_tensor` | 1 (perf case: 3072 × 4096 = 12,582,912 B) | every page is read once by the one core that owns it; no second pass | 0 |
| `output_tensor` | 1 (12,582,912 B) | every page is written once by the same core, from the staged block; never read | 0 |

**Total per tier (perf case):**
- DRAM: 25,165,824 B — the minimal form, the spec's roofline figure.
- Cross-core NoC: 0 B.
- Core-local L1: each byte written once (read landing) and read once (write source); 12,582,912 B
  each way summed over cores.

> Cheapest-traffic split considered: `page` in flat contiguous ranges with coalesced bank runs —
> 0 B delta per tier (every candidate split moves 2 × tensor bytes across DRAM and 0 B cross-core).
> Implemented: the same split. It is the cheapest-traffic split; among the tied splits it is the
> catalog's default transfer shape (coalesced), with bank affinity and bank stagger left as perf
> lamps.
