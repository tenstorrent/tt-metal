# L1 Ledger: g6_fill

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking axes:
`.claude/references/blocking-model.md` and `op_design.md` → Blocking Model → Axes.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_fill_row` | `fill_row_depth` = 1 page of `segment_bytes = face_w * elem_bytes` (64 B for float32) | 1 page. The segment of `e` is the NoC source of every write the core issues, so it stays live from `materialize_fill_row` to `drain_writes`. | `{w_tiles: streams → re-read once per segment, never rewritten; h_tiles: streams → extent 1; segment: streams → re-read for face 0 and face 1; segment_elems: spans → segment_bytes / elem_bytes = face_w}` | `output_tensor.dtype` (Float32 in Phase 0; Int32 in the refinement). Raw 32-bit words written by RISC stores and read by the NoC only. No DEST, unpacker or packer is involved, so the `fp32_dest_acc_en` audit does not apply (dataflow-only payload). | writer (RISC word stores) | writer (NoC write source) | whole kernel | **No other CB exists.** It cannot be removed. `noc_async_write` needs an L1 source, and the source-less `noc_inline_dw_write` does not support DRAM (`noc.h:655`). It cannot be shrunk below one segment without multiplying the write count. |

**Inventory before solve.** One buffer. No reader CB: the input is never read. No compute CB: there is no compute kernel. No output CB: the source is reused for every write. The phase boundary materialize → store is expressed in place, since `store_block` reads the page `materialize_fill_row` wrote. No L1 budget predicate, safety fraction or blocking search exists.

### Symbol table

| Symbol | Bound | Predicate that establishes it |
|--------|-------|-------------------------------|
| `face_w` | 16 | `output_tensor.tile.face_shape[1]`; TILE layout uses 32×32 tiles with 16×16 faces |
| `elem_bytes` | 4 | dtype ∈ {float32, int32} (validate step 1) |
| `segment_bytes` | 64 (TILE); `<= segment_cap` for the ROW_MAJOR refinement | derived from the two above |
| `fill_row_depth` | 1 | Blocking Model → Buffer-depth knobs |
| `N` | unbounded | **does not appear in any capacity expression**: the footprint is independent of the tensor |

### Total per-core footprint

`L1(core) = fill_row_depth * segment_bytes = 1 * 64 B = 64 B` (plus the allocator's CB alignment). This is constant in `N`, in `block_w_tiles` and in the core count. Only `segment_bytes` scales, with dtype width × `face_w`. Bound BB-05: 1,347,776 B per core, met for every N.

## Data-movement budget

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input_tensor` | 0 | Never read: shape, dtype, layout and placement come from host metadata. | none |
| `output_tensor` | 1 (logical segments only) | Each logical face-row segment is written once by exactly one core. Padding rows and faces 2–3 are never written. The tail tile drops its face-1 segment when it holds `<= 16` logical positions. | none |
| `e` (4 B, not a tensor) | 0 | Common RT arg, written by the dispatcher into each core's kernel-config region and materialized locally. | none issued by the op |

Totals per production case: DRAM read 0 B. DRAM written `tensor_segments * 64` B:

| N | DRAM written | writes | logical bytes |
|---|--------------|--------|---------------|
| 512 | 2,048 B | 32 | 2,048 B |
| 256 | 1,024 B | 16 | 1,024 B |
| 207 | 832 B | 13 | 828 B |
| 196 | 832 B | 13 | 784 B |
| 185 | 768 B | 12 | 740 B |

NoC cross-core traffic: 0 B.

> Cheapest-traffic split considered: `w_tiles` across cores with face-row segment writes. Delta vs. every other split: 0 B per tier. The one-segment-per-core and single-core alternatives move identical bytes, and whole-tile writes move 32× more DRAM bytes. Implemented: the same split. The implemented split is also the cheapest.
