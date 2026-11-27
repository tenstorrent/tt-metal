# L1 Ledger: g6_typecast

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes and knobs: `op_design.md` →
Blocking Model. This ledger covers the built regime R1 (`tile_stream_convert`). A deferred regime adds its own
rows when it is built (R2 drops `cb_output_tiles` and keeps one reader→writer CB).

## Inventory

Two buffers are the minimum for R1: the conversion changes page format and page size (Float16_b 2048 B →
Float32 4096 B / Bfp8_b 1088 B / Int32 4096 B), so packing in place into the input CB is impossible, and there is
no intermediate phase to fold (unpack → optional SFPU → pack happens inside one DEST pass). The input CB cannot be
the destination of the pack and the output CB cannot be the reader's landing buffer without changing its format.

## CBs

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_tiles` | `in_buffer_depth · block_tiles` | `block_tiles` — compute waits for one whole block and holds it through that block's DEST pass | `{batch: streams (folded into page), tile_row: streams (folded into page), tile_col: streams (folded into page), page: spans → block_tiles, elem: spans → 1 whole tile per page}` | input dtype; Phase 0 Float16_b. Dataflow payload of the input tensor (input ABI), not a DEST product | reader | compute | whole kernel, every block | Shares with nothing: live for the whole kernel concurrently with `cb_output_tiles` (cross-block pipelining), and differs from it in page format and page size. Capacity > live set by one block: double buffering (`in_buffer_depth = 2`) so the reader lands block k+1 while compute converts block k |
| `cb_output_tiles` | `out_buffer_depth · block_tiles` | `block_tiles` — the writer waits for one whole block and holds it until its writes are flushed | `{batch: streams (folded into page), tile_row: streams (folded into page), tile_col: streams (folded into page), page: spans → block_tiles, elem: spans → 1 whole tile per page}` | output dtype: Float32 (`fp32_dest_acc_en = True`), Int32 (`True`), Bfp8_b (`False`, `bfp8_pack_precise = True`) — width matches DEST in each case; the page format is the output ABI | compute | writer | whole kernel, every block | Shares with nothing: concurrent with `cb_input_tiles` for the whole kernel; different page format and size. Capacity > live set by one block: double buffering (`out_buffer_depth = 2`) so compute packs block k+1 while the writer drains block k |

## Symbol table

| Symbol | Bound | Predicate / source establishing it |
|--------|-------|------------------------------------|
| `block_tiles` | `1 ≤ block_tiles ≤ dest_capacity_tiles` | host `_block_tiles() = min(dest_capacity_tiles, max_core_tiles)`; compute kernel `static_assert(block_tiles ≤ chain_max_block_v)` |
| `dest_capacity_tiles` | ∈ {4, 8} | half-sync DEST (`dst_full_sync_en = False`), lane width 1: 4 when `fp32_dest_acc_en`, else 8 (`dest_helpers.hpp:96-100`); `fp32_dest_acc_en` from the Conversion Pair Table |
| `in_buffer_depth`, `out_buffer_depth` | = 2 (Phase 0) | host constants |
| `in_page_bytes` | ∈ {2048 (bf16, uint16), 4096 (fp32, int32, uint32)} | `input.buffer_page_size()`, input dtype ∈ TARGET |
| `out_page_bytes` | ∈ {4096 (fp32, int32, uint32), 2048 (bf16), 1088 (bfp8_b), 1024 (uint8)} | `output.buffer_page_size()`, output dtype ∈ TARGET |
| `max_core_tiles` | unbounded op dimension (⌈tensor_tiles / num_cores⌉) | appears only inside `min(·)` for `block_tiles`; never in a capacity on its own |

## Total per-core footprint

`in_buffer_depth · block_tiles · in_page_bytes + out_buffer_depth · block_tiles · out_page_bytes`

| Pair class | Value | Bytes |
|------------|-------|-------|
| `bf16→fp32`, `bf16→int32` (Phase 0) | `2·4·2048 + 2·4·4096` | 49,152 |
| `bf16→bfp8_b` (Phase 0) | `2·8·2048 + 2·8·1088` | 50,176 |
| worst TARGET TILE pair (32-bit → 32-bit) | `2·4·4096 + 2·4·4096` | 65,536 |

Both terms scale linearly with `block_tiles`; each term scales linearly with its own buffer depth; page bytes
are fixed per pair. The footprint is independent of the tensor shape, so no golden shape can push the CBs past
the 1,347,776-byte budget. For L1 placement the output tensor's share of each L1 bank is added by the allocator:
`⌈tensor_tiles / num_l1_banks⌉ · out_page_bytes` (largest Phase 0 case P39: `⌈8192 / num_l1_banks⌉ · 1088`,
≈ 70–81 KiB at 110–128 banks). CBs + output share stay ≪ 1,347,776 B.

## Data-movement budget (chosen split: R1, contiguous page ranges)

The boundary is the tensor's interleaved bank (a DRAM bank for `buffer = dram`, a core's L1 bank for `buffer = l1`).

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input | 1 | per-element map: each tile page is consumed by exactly one block on exactly one core; no second pass, no re-read | none for DRAM placement; for L1 placement the single crossing is a NoC read from the page's L1 bank (mesh hop), not a re-read |
| output | 1 | each tile page is packed once and written once by the core that converted it | none for DRAM placement; for L1 placement one NoC write to the page's L1 bank |

Bytes per tier, `tensor_tiles · (in_page_bytes + out_page_bytes)` across the bank boundary, once:

| Case | Placement | Input bytes | Output bytes | Total |
|------|-----------|-------------|--------------|-------|
| Q2185 `(16, 4096, 12)` bf16→fp32 | DRAM | 2048 · 2048 = 4,194,304 | 2048 · 4096 = 8,388,608 | 12,582,912 |
| Q2187 `(1, 1, 3844, 256)` bf16→bfp8_b | DRAM | 968 · 2048 = 1,982,464 | 968 · 1088 = 1,053,184 | 3,035,648 |
| Q2229 `(1, 1, 32, 128)` bf16→int32 | DRAM | 4 · 2048 = 8,192 | 4 · 4096 = 16,384 | 24,576 |
| P39 `(1, 128, 256, 256)` bf16→bfp8_b | L1 | 8192 · 2048 = 16,777,216 | 8192 · 1088 = 8,912,896 | 25,690,112 |

Core-local L1 traffic per tile (every split): the NoC writes the input page into `cb_input_tiles`, the unpacker
reads it, the packer writes `cb_output_tiles`, the NoC reads it out: `2 · in_page_bytes + 2 · out_page_bytes`.
Tile padding moves with its tile (W = 12 → 12/32 of each page is logical); the minimum is per page, not per
logical element.

> Cheapest-traffic split considered: **L1 bank-local (R4)** for `buffer = l1` — delta: removes every mesh traversal
> of the six L1 cases (P22, P31, P39, P46, P50, Q2227: each page still crosses its own bank boundary once, now
> core-locally); for `buffer = dram` every candidate split ties at one crossing per tensor. Implemented:
> **contiguous page ranges (R1)**. Deferred because no host binding exposes the L1 bank→core map
> (`ttnn/cpp/ttnn-nanobind/device.cpp:205-214` exposes only `MemoryView.num_banks`) and R1 already serves all six
> L1 production cases at one crossing per page per tensor. The reader/writer page walk takes
> `(start_page, page_stride, core_pages)` as RT args, so R4 is an RT-arg change plus one core per bank.
