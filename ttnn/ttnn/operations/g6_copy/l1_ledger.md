# L1 Ledger: g6_copy

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes from `op_design.md` →
Blocking Model: `page` (the block axis), `batch` / `tile_row` / `tile_col` (folded into `page`),
`intra_page` (whole page, not split).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_page_staging` | `cb_depth_blocks · block_pages`, where `block_pages = min(core_pages_max, max(1, L1_BUDGET_BYTES // (CB_DEPTH_BLOCKS · page_bytes)))` and `cb_depth_blocks = min(CB_DEPTH_BLOCKS, ceil(core_pages_max / block_pages), L1_BUDGET_BYTES // (block_pages · page_bytes))` | **= capacity.** With one block per core (every golden shape), the whole block is live from the last read landing until the writes flush. With ≥ 2 blocks, block k (being written) and block k+1 (being read) are live at once. No gap to account for | `{page: spans → block_pages × cb_depth_blocks; batch: folded into page (spanned through page); tile_row: folded into page (spanned through page); tile_col: folded into page (spanned through page); intra_page: spans → whole page (page_bytes per slot)}` | The tensor's storage format, never interpreted (float32 in Phase 0; later bfloat16 / bfloat8_b / int32 / uint32 tiles or ROW_MAJOR rows). Audit 2 does not apply: this is a dataflow-only payload that never passes through DEST, and the op has no `fp32_dest_acc_en`. The bytes must stay bit-identical, so any other width would be a contract violation | reader | writer | whole kernel (`load_block` → `store_block`, every block) | **The only CB in the program, so there is nothing to share with. It cannot be removed:** a Tensix NoC read lands only in the issuing core's L1 (dataflow_api.h:529-530), so DRAM → DRAM needs an L1 hop. Pattern 1 (pack into destination): n/a, the destination is DRAM. Pattern 2 (in place): this is the single buffer. Pattern 3 (alias): there is no second buffer. Pattern 4 (fold into DEST): there is no compute |

## Symbol table

| Symbol | Meaning | Bound | Predicate establishing the bound |
|--------|---------|-------|----------------------------------|
| `page_bytes` | `buffer_aligned_page_size()` | TILE: 1088 (bfp8), 2048 (bf16), 4096 (fp32 / int32 / uint32). ROW_MAJOR: `round_up(W · elem, 64)`. Always ≤ `L1_BUDGET_BYTES` | Host refusal when `page_bytes > L1_BUDGET_BYTES` (BB-05); unreachable for TILE; `sub_page_rows` is the deferred regime past it |
| `tensor_pages` | `buffer_num_pages()` | unbounded op dimension | Never enters a capacity directly; only through `core_pages_max` |
| `core_pages_max` | `ceil(tensor_pages / num_cores)` | unbounded | Enters capacity only through `min(·, stream_block_cap)`, so it is bounded by the block clamp |
| `num_cores` | `min(tensor_pages, max_cores)` | ≤ grid cores (110 here) | `split_work_to_cores` over the `max_cores` pool |
| `L1_BUDGET_BYTES` | L1 per core left to the op by the model | 1347776 (constant) | Model environment / BB-05 |
| `CB_DEPTH_BLOCKS` | depth knob before clamping | 2 (constant) | knob |
| `stream_block_cap` | `max(1, L1_BUDGET_BYTES // (CB_DEPTH_BLOCKS · page_bytes))` | 164 fp32 pages; 329 bf16; 619 bfp8 | derived |

**Total per-core footprint:** `L1_op = cb_depth_blocks · block_pages · page_bytes ≤ L1_BUDGET_BYTES`.
- Scales with the `page` block knob (`block_pages`) and the depth knob. Nothing else is allocated: no other CB, no semaphores, no L1 buffers.
- For every golden INPUTS shape, `cb_depth_blocks = 1` and `L1_op = core_pages_max · page_bytes`. The largest values are 114,688 B (float32 (16, 12, 128, 128): 28 × 4096) and 36,864 B (bfloat16 (1, 1, 32, 62080): 1940 tiles over 110 cores = 18 × 2048). Both are far below 1347776.
- The worst case for any shape is `2 · stream_block_cap · page_bytes ≤ 1347776`.

## Data-movement budget

Chosen split: `page` in bank-major contiguous q-ranges over the full grid, with coalesced bank-run
transfers.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input_tensor` | 1 (perf case: 3072 × 4096 = 12,582,912 B) | Every page is read once by the one core that owns its q; there is no second pass, so no re-read | 0 |
| `output_tensor` | 1 (12,582,912 B) | Every page is written once by the same core, from the staged block | 0 |

**Total per tier (perf case):**
- DRAM: 25,165,824 B. This is the minimal form, matching the spec's figure for the roofline.
- Cross-core NoC: 0 B.
- Core-local L1: each byte written once (read landing) and read once (write source), 12,582,912 B each way summed over cores.

> Cheapest-traffic split considered: `page` in bank-major q-ranges with coalesced runs — 0 B delta
> per tier (every candidate split moves 2 × tensor bytes across DRAM and 0 B cross-core).
> Implemented: the same split. It is the cheapest-traffic split, and among the tied splits it
> issues the fewest NoC transactions (4× fewer than per-page transfers for fp32 tiles).
