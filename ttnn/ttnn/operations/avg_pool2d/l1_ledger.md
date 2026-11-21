# L1 Ledger: avg_pool2d (regime `col_split_stream`, Phase 0)

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Target: `quasar` (one cluster's
L1 = 4,194,304 B, shared by its NEOs and DM cores). Thread plan (1, 1, 1): capacities are per
compute thread × C = 1.

Block axes (from `op_design.md`, Blocking Model): `positions` (extent `block_pos_tiles = Ht`),
`channel` (extent `block_ct_tiles`), `batch` (extent `block_images = 1`).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_tiles` | `INPUT_DEPTH_CHUNK_ROWS × dest_chunk_tiles` (Phase 0: 2 × 4 = 8 with 32-bit DEST; 2 × 8 = 16 with 16-bit DEST) | 1 tile for compute (`WaitAndPopPerTile`); up to the capacity in flight on the reader side | `{positions: streams → one tile row of a chunk at a time, channel: streams → window of dest_chunk_tiles, batch: streams → 1 image}` | Float16_b (the input's bfloat16; unpacked to SrcA, never to DEST) | reader (explicit sync: reserve / tagged reads / one barrier / R1 zero fill of rows [v, 32) on the last position tile row / push, one batch per chunk row, cut at the ring's end) | compute | whole program, every block | Capacity exceeds the compute live set by design: the gap is reads in flight (double buffering by chunk row, catalog `double_buffer`). Cannot share with `cb_output_tiles` or `cb_scaler`: all three are live concurrently (the reader streams block k+1 while compute holds the scaler and the writer drains block k). |
| `cb_scaler` | 2 (constant: [full scaler, partial scaler]) | 2 | `{positions: neither — constant (the partial tile encodes only last_tile_valid_rows ∈ [1, 32]), channel: neither — constant, batch: neither — constant}` | Float16_b (the reduce LLK's scaler format, matching the bfloat16 input source; `prepare_reduce_scaler` supports Float16_b/Float32 only) | writer | compute | whole program (filled once, held at the front for every block, popped once at the end) | Cannot alias any other ring: it is resident for the entire program while both other rings cycle. Cannot fold into DEST: the FPU reduce reads it as SrcB. Two tiles, not one: the reduce helper picks the partial scaler by index for the last position row (`with_partial()`, index 1). |
| `cb_output_tiles` | `OUTPUT_DEPTH_CHUNKS × dest_chunk_tiles` (Phase 0: 2 × 4 = 8 / 2 × 8 = 16) | one DEST chunk: `min(dest_chunk_tiles, block_ct_tiles)` tiles packed back to back per chunk | `{positions: neither — reduced away, channel: spans → dest_chunk_tiles per chunk (block_ct_tiles streams over chunks), batch: streams → 1 image}` | Float16_b — the output ABI (bfloat16 result, R8). With `fp32_dest_acc_en=True` the whole sum is complete in 32-bit DEST before this single pack, so no accumulation is truncated at a phase boundary (audit 2 *under* does not apply: this is the final result, not an intermediate). | compute | writer | whole program, every block | Capacity = 2 chunks: the second chunk is double buffering (compute packs chunk k+1 while the writer drains chunk k). Not shared with `cb_input_tiles`: concurrent lifetimes (both cycle every block). Not packed in place into the input ring: different shape (one output tile per `Ht` input tiles) and the input ring has a dataflow producer. |

## Symbol table

| Symbol | Bound | Predicate / source establishing it |
|--------|-------|------------------------------------|
| `dest_chunk_tiles` | ∈ {4, 8, 16} (Phase 0: 4 or 8 with `double_buffer_dest=True`) | host: `(4 if fp32_dest_acc_en else 8) × (1 if double_buffer_dest else 2)`, the rule of `dest_helpers.hpp:95-100`; compute `static_assert(DEST_AUTO_LIMIT == dest_chunk_tiles)` |
| `INPUT_DEPTH_CHUNK_ROWS` | = 2 (host constant) | buffer-depth knob, Phase 0 value |
| `OUTPUT_DEPTH_CHUNKS` | = 2 (host constant) | buffer-depth knob, Phase 0 value |
| `tile_bytes` | = 2048 B (bfloat16 32×32 tile) | `input_tensor.buffer_page_size()` (TILE, bfloat16: Phase 0 gating) |
| `Ht`, `block_ct_tiles`, `core_ct_tiles`, `tensor_ct_tiles`, `last_tile_valid_rows` | do **not** appear in any capacity | — (streaming rings; the scaler pair is constant) |
| per-core budget | `ttnn.get_max_worker_l1_unreserved_size() − l1_reserve_bytes`; ≤ 4,194,304 B on this target (value of the query not probed) | no predicate spends it: the footprint below is bounded by a constant far below any plausible reserve |

## Total per-core footprint

```
footprint = tile_bytes × ( INPUT_DEPTH_CHUNK_ROWS × dest_chunk_tiles     // cb_input_tiles  — scales with the input depth knob and DEST mode
                         + 2                                             // cb_scaler       — constant
                         + OUTPUT_DEPTH_CHUNKS × dest_chunk_tiles )      // cb_output_tiles — scales with the output depth knob and DEST mode
```

Phase 0 on this target: 32-bit DEST: 2048 × (8 + 2 + 8) = 36,864 B; 16-bit DEST: 2048 × (16 + 2 +
16) = 69,632 B. No term scales with any op dimension (H, W, C, N) or with the block extents: the
rings stream, and the block's positions accumulation lives in DEST. Inventory check: three rings is
the minimum for this scheme — one ring per kernel-to-kernel stream (reader→compute data,
writer→compute constant, compute→writer result); no intermediate exists to pack into a destination
or transform in place, because the reduce writes its result straight into the output ring.

## Data-movement budget

Tile bytes B = 2048. One image, `Ht × tensor_ct_tiles` input tiles. Model call (7×7×2048, 32
clusters): Ht = 2, `tensor_ct_tiles` = 64.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input | 1 for a DRAM input (`Ht·C_t·B`; model 262,144 B); 0 for an L1 input | each tile is read exactly once, by the one cluster that owns its channel column; nothing is re-read because the dependent axis stays whole on one cluster and accumulates in DEST | for an L1 input (interleaved or width-sharded) the single read is an L1→L1 NoC crossing of the same bytes (model: 262,144 B); no combine traffic |
| scaler pair | 0 | built in each engaged cluster's own L1 by its writer (`prepare_reduce_scaler`), not read from anywhere | 0 (2 × B of local L1 writes per engaged cluster) |
| output | 1 for a DRAM input's DRAM output (`C_t·B`; model 131,072 B, of which 1/32 is real data — TILE padding rows); 0 for an L1 output | written once, by the cluster that reduced it | for an L1 output, one L1 write per tile from the owning cluster (`C_t·B`) |

Totals per tier, model call: DRAM-interleaved input → DRAM 262,144 B read + 131,072 B written,
cross-core 0. L1 (interleaved or width-sharded) input → DRAM 0, NoC 262,144 B input + 131,072 B
output, combine traffic 0.

> Cheapest-traffic split considered: `shard_in_place_width` (channel split taken from the caller's
> width shards, zero copy) — delta for a width-sharded input: −1 L1→L1 crossing of the input
> (model: −262,144 B on the NoC), DRAM unchanged (0). For interleaved inputs the implemented split
> is itself the cheapest (input once, output once, no combine). Implemented: `col_split_stream`
> (channel tiles over the full grid). Deferred because the implemented split already serves every
> width-sharded shape on any shard grid (R10), and on the specification's traced call the bytes it
> saves buy away occupancy (the model's input has 2 shards; the zero-copy split would reduce on 2
> of 32 clusters) — an explicit bytes-for-occupancy trade recorded in the regime table, with the
> predicate (`num_shard_cores ≥ min(tensor_ct_tiles, grid_cores)`) under which the zero-copy
> regime is strictly better. The structure keeps it reachable: the compute kernel and its block
> operation are unchanged; only the reader (credit-only producer on a borrowed ring) and the host
> split change.

## Implementation note (implementer, Phase 0)

`cb_input_tiles` is **explicit sync** (opted out on the reader), not implicit as the design's Dataflow
Strategy row has it. The
reader zeroes rows [last_tile_valid_rows, 32) of each last-position-row tile in the ring slot after its read
lands and before it pushes it: the fallback of the design's Key Risks row. Capacity, live set and lifetime
are unchanged; the zero fill is a local L1 write (overlay iDMA on Quasar), 0 NoC bytes, so the
data-movement budget below is unchanged. `cb_scaler` keeps the partial scaler (harmless with zeroed
padding).
