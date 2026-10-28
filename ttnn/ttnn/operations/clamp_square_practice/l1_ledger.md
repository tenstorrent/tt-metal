# L1 Ledger: clamp_square_practice (regime `interleaved_stream`)

The schema and audits are defined in `.claude/references/l1-footprint-discipline.md`. The block axes are named in `op_design.md` → Blocking Model:
- `tile` axis (flattened `batch · tile_row · tile_col`): extent `block_tiles`

`fp32_dest_acc_en = False` (bf16 in and out, SFPU-only chain), so DEST is 16-bit and every compute-written page is Float16_b (audit 2: no wide page under narrow DEST).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_tiles` | `input_depth_blocks · block_tiles` (Phase 0: `2 · 8 = 16`) | `block_tiles` being consumed by compute + up to `block_tiles` being filled by the reader = `2 · block_tiles` at peak | `{tile: spans → block_tiles (× input_depth_blocks)}` | Float16_b (input dtype bf16) | reader | compute | per block: `load_block` → end of `clamp_square_block` DEST window | **Cannot share with `cb_output_tiles`.** Their lifetimes are concurrent: pipelining keeps block `k` in the output while block `k+1` sits in the input. Packing in place into `cb_input_tiles` would make the writer a second consumer of a reader-produced CB, which violates CB ownership. Capacity is twice the compute live set to give double buffering (overlap of fetch and compute, catalog `double_buffer`). |
| `cb_output_tiles` | `output_depth_blocks · block_tiles` (Phase 0: 16) | `block_tiles` being drained by the writer + `block_tiles` being packed = `2 · block_tiles` | `{tile: spans → block_tiles (× output_depth_blocks)}` | Float16_b (DEST 16-bit, output bf16) | compute | writer | per block: `clamp_square_block` pack → `store_block` | **Cannot share with `cb_input_tiles`** for the same concurrent-lifetime and ownership reasons. Capacity of 2× is for double buffering (overlap of pack and DRAM write). |

**Inventory before solve.** There are two CBs, which is the floor for a reader → compute → writer stream with separate producer and consumer kernels on each side. The Clamp → Square phase boundary is **not** storage: both are SFPU ops on DEST in one chain window (reuse pattern 4, fold into DEST). There is no scaler or constant CB, because the bounds are runtime args. No budget predicate, safety fraction or blocking search exists.

## Symbol table

| Symbol | Bound | Predicate / source establishing it |
|--------|-------|------------------------------------|
| `tile_bytes` | 2048 | `ttnn.tile_size(ttnn.bfloat16)`. Phase 0 dtype gate. |
| `block_tiles` | `1 ≤ block_tiles ≤ min(8, core_tiles_g1)` | Host clamp `min(BLOCK_TILES_DEFAULT=8, dest_limit(fp32_dest_acc_en=False, dst_full_sync_en=False)=8, core_tiles_g1)`. The FullBlock DEST cap is at `chain.hpp:504-508`. |
| `input_depth_blocks` | 32 (Perf 1; was 2). Capacity = `min(32·block_tiles, ceil(core_tiles_g1/block_tiles)·block_tiles)` ≤ 256 tiles. | Host constant; Perf 1 measured that deep CBs hide compute under the DRAM-bound DM |
| `output_depth_blocks` | 32 (Perf 1; was 2). Same cap as the input CB. | Host constant |
| `core_tiles_g1` | `ceil(tensor_tiles / num_cores)`. This does not appear in any capacity, only in the clamp. | `ttnn.split_work_to_cores` |

No op dimension (`tensor_tiles`, `H`, `W`) appears in any capacity, so the footprint is independent of tensor size.

## Total per-core footprint

```
L1_cb(block_tiles, input_depth_blocks, output_depth_blocks) =
      input_depth_blocks  · block_tiles · tile_bytes     # cb_input_tiles  — scales with block_tiles × input depth
    + output_depth_blocks · block_tiles · tile_bytes     # cb_output_tiles — scales with block_tiles × output depth
```

Phase 0: `2·8·2048 + 2·8·2048 = 65 536 B` (64 KiB) per core. **Perf 1:** each CB is `min(32·block_tiles, assignment rounded up to a block)` tiles, so the worst case is `2·256·2048 = 1 MiB` per core, independent of tensor size (focus (1,1,4096,4096): 2·144 tiles = 576 KiB). That is about 5% of usable L1 for any input size. The max over the knob ranges (`4·8·2048·2 = 128 KiB`) also fits trivially.

---

## Data-movement budget

`T = tensor_tiles · 2048 B` (bf16), with `tensor_tiles = prod(shape[:-2]) · ceil(H/32) · ceil(W/32)`.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input | 1 (`T`) | Each tile is read by exactly one core, exactly once. There is no reuse, so there is nothing to keep resident. | 0 |
| output | 1 (`T`) | Each tile is packed once and written once | 0 |
| `min_val`, `max_val` | 0 | Runtime args (2 × 4 B per core, delivered with the program), not tensors | 0 |

Totals per tier: DRAM `2T`. NoC cross-core `0`. Core-local L1: one unpack read and one pack write per tile (`2T`). There is no intermediate round trip.

> Cheapest-traffic split considered: flattened `tile` axis, contiguous per-core page ranges — `2T` DRAM, 0 cross-core (this is the minimum: each input once, each output once). Implemented: the same split. The implemented split is the cheapest. Every other candidate (tile-row split, tile-column split) moves the same `2T` and loses on occupancy or transfer fragmentation, and no dependent-axis split exists.
