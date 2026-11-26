# L1 Ledger: g6_to_memory_config

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking decisions: `op_design.md` → Blocking Model.

## Circular buffers

The four owner regimes (`shard_pull`, `shard_push`, `bank_pull`, `bank_push`) allocate **no CB**: a pulled page lands at its final address in the output's own L1 region, a pushed page leaves from the input's own L1 region. Inventory before solve: the only buffer in the op is the staging buffer of `staged`, which exists because a NoC transaction cannot move DRAM → DRAM.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_staged_pages` (`staged` only) | `staged_buffer_depth · staged_block_pages` = 2 · 16 = 32 | `2 · staged_block_pages` at peak — the block the writer drains plus the block the reader fills; capacity == live set | `{page_linear: spans → staged_block_pages per depth slot, streams over core_pages; page_row: flattened into page_linear; page_col: flattened into page_linear; intra_tile: spans → 1 whole page per slot (atomic); mover: n/a — staged uses one reader + one writer}` | input dtype (`Float16_b` / `Bfp8_b` / `Bfp4_b`): dataflow-only payload, never unpacked or packed, so DEST width (`fp32_dest_acc_en`) does not apply — audit 2 answered by "dataflow-only payload" | reader (staged) | writer (staged) | whole `staged` kernel | nothing to share with: it is the op's only CB and exists only in the `staged` regime; depth 2 is the explicit pipelining decision (reader fills block k+1 while writer drains block k) |

## L1 counted against the BB-05 bound that is not a CB

The bound counts the output's own L1 on each core. These are allocations of the output tensor, not op buffers; they are listed so the total is closed-form.

| Resident | Bytes per core | Present when | Axis accounting |
|----------|----------------|--------------|-----------------|
| Output L1 shard | `shard_tile_rows(out) · shard_tile_cols(out) · aligned_page_size` | output is L1-sharded (`shard_pull`, or `shard_push` between two L1-sharded placements) | `{page_row: spans → shard_tile_rows(out); page_col: spans → shard_tile_cols(out); page_linear: n/a; intra_tile: whole}` |
| Output L1-interleaved bank region | `ceil(tensor_pages / NB) · aligned_page_size` | output is L1-interleaved (`bank_pull`, or `shard_push` to L1 interleaved) | `{page_linear: spans → ceil(tensor_pages/NB) (cyclic share); page_row, page_col: flattened; intra_tile: whole}` |
| Input (any placement) | not counted (the model's tensor, per the spec) | always | — |

## Symbol table

| Symbol | Meaning | Bound | Predicate establishing the bound |
|--------|---------|-------|----------------------------------|
| `aligned_page_size` | bytes per tile page | 2048 (bf16), 1088 (bf8b), 576 (bf4b) | dtype ∈ TARGET; each is a multiple of the 16 B L1 and 64 B DRAM alignment |
| `staged_block_pages` | staged block extent | 16 (Phase 0 constant `STAGED_BLOCK_PAGES`) | host constant; CT arg; clamped to `core_pages` |
| `staged_buffer_depth` | staged depth | 2 (`STAGED_BUFFER_DEPTH`) | host constant |
| `NB` | L1 banks = worker cores | 110 on this board (`l1_num_banks == compute_with_storage_grid` checked per device) | allocator: one L1 bank per worker core (`l1_banking_allocator.cpp:96-111`) |
| `tensor_pages` | `prod(leading)·ceil(H/32)·ceil(W/32)` | unbounded by contract; golden max 81920 (bf4b, DRAM→DRAM, no L1 term) and 10240 for an L1 output (Qwen 2078) | enters L1 only through `ceil(tensor_pages/NB)`; bounded by the BB-05 refusal predicate below |
| `shard_tile_rows(out)`, `shard_tile_cols(out)` | target shard extent in pages | caller's shard spec | bounded by the BB-05 refusal predicate below |
| `L1_BUDGET` | per-core L1 the model leaves to the op | 1347776 B | spec (BB-05) |

## Total per-core footprint

```
op_l1_bytes_per_core =
      [out is L1-sharded]      · shard_tile_rows(out) · shard_tile_cols(out) · aligned_page_size
    + [out is L1-interleaved]  · ceil(tensor_pages / NB) · aligned_page_size
    + [regime == staged]       · staged_buffer_depth · staged_block_pages · aligned_page_size
```

- Scales with the target placement only (shard extent or `tensor_pages/NB`); the owner regimes add **0** bytes of op buffer for any shape.
- The only knob-scaled term is the `staged` CB (`staged_buffer_depth · staged_block_pages`): 65536 B bf16, 34816 B bf8b, 18432 B bf4b at Phase 0 values; it never coexists with an L1 output term (`staged` ⇔ both ends DRAM).
- Host refusal (BB-05): `ValueError` when `op_l1_bytes_per_core > 1347776`, evaluated before the output is allocated.
- Golden INPUTS maximum: Qwen 2078, `bank_pull`, `ceil(10240/110) · 2048 = 192512 B` (14.3 % of the budget); next Qwen 2075 163840 B; the largest L1 shard output is 17 pages (34816 B, Qwen 2064/2092/2127/2161). Every golden shape fits; the refusal concerns shapes outside the golden INPUTS only.

## Data-movement budget

Per regime, `B = tensor_pages · aligned_page_size` bytes:

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input (`shard_*`, `bank_*`) | 1 if in DRAM, else 0 | each page read once by the core that owns its source or target slot; nothing is re-read (each core's slice is disjoint and every transfer is one run) | 1 NoC crossing per page (B bytes total); 0 extra |
| output (`shard_*`, `bank_*`) | 1 if in DRAM, else 0 | written once, straight from the mover's L1 | included above (the same single crossing) |
| input (`staged`) | 1 | read once into `cb_staged_pages` | DRAM → staging core: B bytes |
| output (`staged`) | 1 | written once from `cb_staged_pages` | staging core → DRAM: B bytes |

Totals per tier: DRAM `B · ([in DRAM] + [out DRAM])`; NoC `B` (owner regimes) or `2B` (`staged`, the structural minimum for DRAM → DRAM); core-local L1 `B` (the owner's own side of each transfer). Production extremes: Qwen 2078 `bank_pull` B = 20971520 B, DRAM 1×B, NoC 1×B; PERF FOCUS Qwen 2183 `staged` B = 47185920 B, DRAM 2×B, NoC 2×B.

> Cheapest-traffic split considered: owner moves (the core holding the page in L1 moves it) — 1 NoC crossing and no staging per page wherever an L1 end exists; any-core staging would add B bytes of L1→L1 traffic and a `2·block·page` buffer per core. Implemented: owner moves (`shard_pull` / `shard_push` / `bank_pull` / `bank_push`), plus `staged` only for DRAM → DRAM where 2 crossings is the minimum. The implemented split **is** the cheapest; nothing is deferred on traffic grounds (`helper_fanout` and `staged_coalesced` are deferred for issue rate and transaction count, not bytes).
