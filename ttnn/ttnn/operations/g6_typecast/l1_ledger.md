# L1 Ledger: g6_typecast (regime `tile_convert`, Phase 0)

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes from
`op_design.md` → Blocking Model: `batch`, `tile_row`, `tile_col` (all three folded into `tile`),
`tile`, `tile_elem`.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_tiles` | `pipeline_depth · block_tiles` | `pipeline_depth · block_tiles` — at the peak the reader has filled slot *k+1* while compute still holds slot *k* | `{batch: folded → tile, tile_row: folded → tile, tile_col: folded → tile, tile: spans → block_tiles per slot × pipeline_depth slots, streams over num_blocks_this_core, tile_elem: spans → one whole tile per page}` | input dtype; Phase 0 bfloat16 (2048 B). Dataflow-landed input page that never passes through DEST, so audit 2 does not apply | reader | compute | whole kernel | none. Not aliasable with `cb_output_tiles`: concurrent lifetimes (compute unpacks block *k* from it while packing block *k* into `cb_output_tiles`, and the slots pipeline). In-place pack impossible: page size and format differ (2048 B bf16 vs 4096 B f32/i32 or 1088 B bf8b) |
| `cb_output_tiles` | `pipeline_depth · block_tiles` | `pipeline_depth · block_tiles` — compute packs slot *k+1* while the writer drains slot *k* | `{batch: folded → tile, tile_row: folded → tile, tile_col: folded → tile, tile: spans → block_tiles per slot × pipeline_depth slots, streams over num_blocks_this_core, tile_elem: spans → one whole tile per page}` | output dtype: Float32 (with `fp32_dest_acc_en=True`), Int32 (with `fp32_dest_acc_en=True`), Bfp8_b (with 16-bit DEST). The width follows the output ABI (output dtype argument), and DEST width matches in every pair | compute | writer | whole kernel | none. Not aliasable with `cb_input_tiles` (above). A packer write straight to the output tensor is impossible: the output pages live in other banks across the NoC |

Capacity equals live set for both CBs. The excess over one block is the double buffer
(`pipeline_depth = 2`), which buys read/convert/write overlap.

## Symbol table

| Symbol | Bound | Predicate that establishes it |
|--------|-------|-------------------------------|
| `pipeline_depth` | 2 (Phase 0) | `Plan` constant (single source) |
| `block_tiles` | `1 ≤ block_tiles ≤ min(core_tiles_max, block_tiles_l1_cap)`; Phase 0 = `min(ceil(core_tiles_max / pipeline_depth), block_tiles_l1_cap)` | `Plan` rule; the host raises `RuntimeError` if `block_tiles_l1_cap < 1` (no golden INPUTS shape reaches it) |
| `core_tiles_max` | `ceil(tensor_tiles / num_cores)`; unbounded in `tensor_tiles` | does not size any CB directly. It enters only through `block_tiles`, which is clamped by `block_tiles_l1_cap` |
| `block_tiles_l1_cap` | `floor((L1_OP_BUDGET − out_bank_bytes) / (pipeline_depth · (in_page_bytes + out_page_bytes)))` | BB-05 budget |
| `in_page_bytes` | 2048 (bf16 tile, Phase 0); ≤ 4096 for every TARGET TILE source | `dtype ∈ SUPPORTED` |
| `out_page_bytes` | 4096 (float32, int32) or 1088 (bfloat8_b) | `output_dtype ∈ SUPPORTED` |
| `out_bank_bytes` | `ceil(tensor_tiles / num_l1_banks) · out_page_bytes` if the output is in L1, else 0. INPUTS max: P39, 64 tiles × 1088 B = 69,632 B with 130 banks | the output is the op's own allocation, in the input's buffer type |
| `L1_OP_BUDGET` | 1,347,776 B per core | model contract (BB-05) |

## Total per-core footprint

```
op_L1 = pipeline_depth · block_tiles · (in_page_bytes + out_page_bytes)   // scales with block_tiles × pipeline_depth
      + out_bank_bytes                                                    // scales with tensor_tiles / num_l1_banks (L1 output only)
      ≤ L1_OP_BUDGET
```

Phase 0 maxima, with block extents at a 13×10 grid (`op_design.md` → Work Distribution):
- P39 (L1): 2·32·(2048 + 1088) + 69,632 = 270,336 B.
- Q2185 (DRAM): 2·8·(2048 + 4096) = 98,304 B.
- Decode shapes (1 tile per core): ≤ 12,288 B.

With an 11×10 grid, P39 uses block 38, which gives 238,336 + ≈81,600 B. Every golden shape stays
below 25 % of the budget. A shape with unbounded rows saturates at `block_tiles_l1_cap`, so the CBs
never grow with the op size.

## Data-movement budget (chosen split: contiguous page ranges, `tile_convert`)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input_tensor` | 1 (DRAM placement) / 0 (L1 placement: it crosses the L1-bank tier once) | streamed once. No second pass and no reuse (elementwise; no reduction, no shared operand) | DRAM placement: none beyond the transfer itself. L1 placement: every page crosses the NoC once, from its bank to the converting core |
| `output_tensor` | 1 (DRAM) / 0 (L1: one crossing of the L1-bank tier) | written once, in the input's buffer type | L1 placement: every page crosses the NoC once, from the converting core to its bank |

Totals per tier: `tensor_tiles · (in_page_bytes + out_page_bytes)` on the tensor's tier.
- Q2185 (DRAM): 4 MiB in + 8 MiB out = 12 MiB of DRAM.
- P39 (L1): 16 MiB in + 8.5 MiB out = 24.5 MiB of cross-core NoC, 0 DRAM.

> DRAM placement: the implemented split is the cheapest. Each tensor crosses DRAM exactly once,
> which is the minimum.
>
> L1 placement: the cheapest-traffic split considered is `l1_bank_affine` (owner computes on its
> own bank's pages). Delta: −`tensor_tiles · (in_page_bytes + out_page_bytes)` cross-core NoC
> bytes (P39: −24.5 MiB), and DRAM is 0 either way. Implemented: contiguous page ranges.
> Deferred because the built `tile_convert` already serves every L1 shape, and the bank-affine
> assignment needs the L1-bank → logical-core map. The per-core page walk
> (`first_page + k·page_stride`) keeps it reachable as a host-side runtime-arg change with
> unchanged kernels. One experiment settles it: P39 with `page_stride = num_l1_banks`,
> `first_page = own bank`, against the built split.
