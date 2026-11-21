# L1 Ledger: typecast

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Target: `quasar` (worked values); predicates query the device. Blocking axes (from `op_design.md`): `tile` (flat page index; the leading / H-tile / W-tile axes are folded into it and carry no separate extent). Thread plan (R, C, W) = (1, 1, 1).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_tiles` | `buffer_depth · block_tiles · C` (entry = `in_page`) | `buffer_depth · block_tiles` per NEO: compute holds block b while the reader fills block b+1 | `{tile: spans → block_tiles (× buffer_depth for overlap)}`; folded leading/H/W: streams (inside `tile`) | input dtype: Float16_b (2,048 B) or Float32 (4,096 B). A dataflow payload as read from DRAM; its width is the input's, not DEST's — the Float32 page under 32-bit DEST is required (UnpackToDest of the full 32 bits) | reader | compute | whole program (kernel start → end) | none: concurrent with `cb_output_tiles` for the whole pipeline (blocks b and b+1 in flight at once), and different page format/size; capacity − one block = the double-buffer slot (buffer_depth, a stated overlap decision). Pack-into-destination / in-place are impossible: input and output formats differ (output page ≠ input page for 2 of 4 pairs) and the input ring has a dataflow producer |
| `cb_output_tiles` | `buffer_depth · block_tiles · C` (entry = `out_page`) | `buffer_depth · block_tiles` per NEO: writer drains block b while compute packs block b+1 | `{tile: spans → block_tiles (× buffer_depth)}`; folded leading/H/W: streams | output dtype: Float16_b or Float32. Float32 only when the output tensor is float32 (output ABI); Float16_b pages are packed from a 32-bit DEST by the packer's narrowing, which is the conversion itself | compute | writer | whole program | none: concurrent with `cb_input_tiles` (pipelined), different format; double-buffer slot as above. Cannot be the input ring transformed in place (format/size differ; two kernels would consume it) |

No scaler, mask, scratch or intermediate buffers: the conversion is unpack → DEST → pack within one chain, folded entirely into DEST (Rule 3.4), so the two rings are the minimum inventory (one source, one sink).

## Symbols

| Symbol | Bound | Predicate / source |
|--------|-------|--------------------|
| `block_tiles` | `1 ≤ block_tiles ≤ block_tiles_cap` | `block_tiles = clamp(ceil(core_tiles_max / buffer_depth), 1, block_tiles_cap)` (host) |
| `block_tiles_cap` | `min(dest_tiles, floor(DFB_RING_MAX_BYTES / (C · buffer_depth · max(in_page, out_page))), floor(l1_budget / (C · buffer_depth · (in_page + out_page))))`; quasar: **4** (the DEST term binds: 32-bit DEST, half sync) | `dest_tiles` from `ENABLE_32_BIT_DEST` / `DOUBLE_BUFFER_DEST` (implementer, see note below); ring cap from the target page; L1 term from the query |
| `buffer_depth` | 2 | module constant |
| `C` | 1 (plan (1,1,1)); legal {1,2,4} | `THREAD_PLAN` constant, validated against the target page's rules |
| `in_page`, `out_page` | 2,048 or 4,096 B (Phase 0 dtypes) | `buffer_page_size()` of the input and output tensors |
| `core_tiles_max` | `ceil(tensor_tiles / num_cores)`; unbounded in the tensor size — **not** used in any capacity except through `block_tiles`, which is clamped | `split_work_to_cores` |
| `l1_budget` | `ttnn.get_max_worker_l1_unreserved_size() − l1_reserve_bytes` = 3,963,328 − 65,536 = 3,897,792 B on quasar | run-time query |
| `DFB_RING_MAX_BYTES` | 1,048,560 B | target page *Cluster model* |

## Total per-core footprint

`L1_rings = C · buffer_depth · block_tiles · (in_page + out_page)`

- scales linearly with `block_tiles`, `buffer_depth` and `C`; constant in the tensor size (block clamped).
- worked (quasar): Phase-0 calls (`block_tiles = 1`): bf16→fp32 = 1·2·1·6,144 = 12,288 B; fp32→fp32 = 16,384 B. Largest INPUTS shape `[1,1,2048,256]` fp32→fp32 (`block_tiles = 4`): 65,536 B. At the cap (`block_tiles = 4`, C = 4, fp32→fp32): 4·2·4·8,192 = 262,144 B.

> Implementer change (DEST cap). The design's Mechanism-caps row "DEST section: none on `block_tiles`" does not hold: `eltwise_chain` clamps a `PerBlockSize` block to its DEST capacity at run time (`ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.inl:3345-3349`) and then waits / pops / reserves / pushes the rings in windows of that clamped size. With a ring of `buffer_depth · block_tiles` entries and `block_tiles` above the DEST section (4 tiles at 32-bit DEST, half sync), a window straddles the ring end. `block_tiles` is therefore capped at `dest_tiles`, and the compute kernel `static_assert`s it against `DEST_AUTO_LIMIT`.

## Data-movement budget

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input_tensor` | 1 | each tile is read once by the one cluster that owns its index (contiguous split); nothing is re-read — no second pass, no shared operand | none |
| output | 1 | each tile written once by its owner | none |

Totals per call: DRAM (or L1-interleaved banks for an L1 input/output) = `tensor_tiles · (in_page + out_page)` bytes (Phase-0 call 1: 8 · 6,144 = 49,152 B; call 2: 16 · 6,144 = 98,304 B); cross-core NoC = 0; core-local L1 = each tile written once into a ring and read once by unpack, written once by pack and read once by the writer.

> Cheapest-traffic split considered: flat contiguous tile ranges (S1) — 0 bytes above the minimum on every tier. Implemented: S1. It is also the cheapest: every candidate split of this elementwise op moves the same minimum bytes (no dependent axis, no shared operand); the round-robin split ties and loses only on argument simplicity.
