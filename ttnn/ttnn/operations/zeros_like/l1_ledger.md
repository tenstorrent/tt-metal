# L1 Ledger: zeros_like

Schema: `.claude/references/l1-footprint-discipline.md`. Target: `quasar` (32 clusters, 4 MiB L1 per
cluster) on ttsim. Blocking axes (from `op_design.md`): `output_page` (block `pages_per_barrier`;
`batch`, `tile_row`, `tile_col` folded into it) and `page_chunk` (`zero_src_bytes`).

**Inventory.** One buffer. The op reads nothing and computes nothing, so there is no input ring, no
intermediate and no output ring: the only thing a write needs in L1 is a source of zero bits, and
one page of them serves every write because the source is never modified after it is zeroed.
Rule 3 patterns: *pack into destination* — the zero page is written straight to the output buffer
(no staging); *transform in place* / *alias* / *fold into DEST* — nothing to transform, nothing else
to alias with, no DEST. No budget predicate, safety fraction or blocking search exists.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `zero_page` (ScratchpadSpec, not a ring; one "page" = `zero_src_bytes`) | `W` slices of `zero_src_bytes`, `zero_src_bytes = min(align_up(page_bytes, l1_alignment), ZERO_SRC_MAX_BYTES)`; Phase 0: `W = 1` → 1 × 2048 B (bf16 TILE) | `W` slices — every slice is read by that thread's in-flight writes for the whole kernel, so capacity = live set | `{output_page: streams → reused for every page (does not scale with pages_per_barrier or num_pages_this_core), batch/tile_row/tile_col: streams (folded into output_page), page_chunk: spans → zero_src_bytes}` | none — raw zero bits; a DM-only payload that never passes through DEST (audit 2 does not apply: no `fp32_dest_acc_en`, no compute) | writer (`zero_source`, iDMA zero) | writer (`store_block`, NoC write source) | whole writer kernel | No other buffer exists to share with. Its depth stays 1 per thread: the bytes are read-only after `zero_source`, so in-flight writes never conflict with a refill and a second copy would overlap nothing |

## Symbol table

| Symbol | Bound | Established by |
|--------|-------|----------------|
| `page_bytes` | TILE: 2048 (bf16) / 4096 (fp32); ROW_MAJOR: `W_elems · elem_size`, unbounded above | `out.buffer_page_size()` (host); no predicate needed because `zero_src_bytes` clamps it |
| `ZERO_SRC_MAX_BYTES` | 65536 (constant) | host constant; multiple of DRAM alignment (64) so chunk offsets stay aligned |
| `zero_src_bytes` | ≤ 65536 | `min(...)` above |
| `l1_alignment` | 16 on this target | `ttnn.get_l1_alignment()` |
| `W` (writer threads) | 1 in Phase 0; ≤ 6 | `THREAD_PLAN`; work-unit validation (`work_unit.cpp:82-87`) |
| `pages_per_barrier` | ≤ 256 | host: `max(1, min(PAGES_PER_BARRIER, TRID_COUNTER_CAP // (chunks_per_page · packets_per_chunk)))` (`block_parameters`, the trid cap); kernel: each block's extent is `min(pages_per_barrier, pages left in the core's range)`. Does not appear in any capacity |
| L1 budget | `ttnn.get_max_worker_l1_unreserved_size() - l1_reserve_bytes` (reserve 0); value on this target not recorded on the target page, ≤ 4,194,304 B | run-time query; host asserts `W · zero_src_bytes ≤ budget` (always true: ≤ 6 × 64 KiB) |

## Total per-core footprint

`footprint = W · zero_src_bytes` bytes — scales with the writer-thread count and with the page size
(dtype, layout) up to the 64 KiB clamp; constant in `pages_per_barrier`, in the per-core page count
and in every op dimension.

| Phase 0 (bf16, TILE) | float32 TILE | ROW_MAJOR, W = 8192 fp32 | Any shape, W = 6 threads (cap) |
|----------------------|--------------|--------------------------|--------------------------------|
| 2048 B | 4096 B | 32768 B | ≤ 393216 B |

## Data-movement budget

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `tensor` (input) | 0 | not bound to any kernel: only its spec is read on the host (R1) | 0 |
| `out` (output) | 1 (written once) | each page is written exactly once by exactly one core, from a zero page generated in that core's L1 | 0 |
| zero page (constant) | 0 | generated locally by iDMA zero mode, once per core per kernel | 0 (no multicast: generation is free per core) |

Per tier, for one call: DRAM `num_pages · page_bytes` written, 0 read (bf16 TILE: `2048 · num_pages`
B; largest INPUTS shape `(1024,1024)` / `(2,512,1024)`: 1024 pages = 2 MiB). Cross-core: 0 B.
Core-local: `W · zero_src_bytes` zeroed per engaged core, plus one L1 read of the zero page per
write by the NoC.

> Cheapest-traffic split considered: `output_page` over the full grid in contiguous ranges — 0 B
> read, `num_pages · page_bytes` written to DRAM, 0 B cross-core. Implemented: the same split. It
> is the cheapest: no split can write fewer bytes than the output or read fewer than zero.
> (`fill_bank_runs`, deferred in the regime table, moves the same bytes in fewer transactions; it
> is not a traffic saving.)
