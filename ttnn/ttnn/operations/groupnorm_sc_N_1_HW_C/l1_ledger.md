# L1 Ledger: groupnorm_sc_N_1_HW_C

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes (from `op_design.md` Blocking Model): `N` (image), `HW` (tile rows), `C` tiles (channel tiles), `G` (group slots). Page formats follow `fp32_dest_acc_en = True`: every compute-produced intermediate is `Float32`; only the input/output staging keeps the tensor dtype and only gamma/beta keep the affine dtype (they are consumed, not produced, by DEST). Refinement 4: `dtype ∈ {bf16, fp32, bf8b}` and `affine_dtype ∈ {bf16, fp32, bf8b}` — `TB`/`TA` follow `ttnn.tile_size`, the intermediates stay `Float32` (DEST is fp32 — `fp32_dest_acc_en=False` is refused by the op file), and bf8b weights are decoded by the reader into a **bf16** rows CB (`TA = 2048` for bf8b weights: a block format cannot be lane-gathered, and bf8b's 7-bit mantissa is exact in bf16).

Symbols: `TB = tile_bytes(dtype)` (2048 bf16 / 4096 fp32 / 1088 bf8b), `T4 = 4096`, `TA = tile_bytes(rows dtype)` (= affine dtype, except bf16 for bf8b weights), `K = block_c_tiles = core_c_tiles`, `H = block_hw_tiles`, `Q = chunk_hw_tiles`, `D = STREAM_DEPTH`, `Ng = num_group_tiles`, `GT = gather_tiles`.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_tiles` | resident: `core_hw_tiles·K`; streaming: `D·Q·K` | resident: `core_hw_tiles·K` (all three passes re-read it); streaming: `Q·K` | `{N: streams → 1 image, HW: resident spans → core_hw_tiles / streaming streams → Q, C: spans → K, G: —}` | input dtype (bf16) | reader (TILE) / compute tilize (RM) | compute | resident: image round; streaming: per block | none — concurrent with every other CB during passes; capacity > live set on the streaming path is the `D`-deep double buffer (overlap) |
| `cb_input_sticks` (RM only) | `D·K` (tile-sized pages = one 32-stick tile-row of `K·64 B` slices per `K` pages) | `K` | `{N: streams, HW: streams → 1 tile-row, C: spans → K, G: —}` | input dtype | reader | compute (tilize) | per tile-row of a load | none — concurrent with `cb_input_tiles` (tilize writes it while sticks arrive); `D` for reader/tilize overlap |
| `cb_scaler` | 1 | 0 | `{N: —, HW: —, C: —, G: —}` constant | `Float16_b` | nobody (Refinement 4: the Phase-0 `prepare_reduce_scaler` fill was dead work — `reduce<..., AccumulateViaAdd>` waits the scaler only for a partial mask or the `CopySeedZeroPair` reload, neither used here — and its ~300 B of reader code was what kept the K = 2 staged RM `hw_mask` shards out of the kernel-config ring in `--dev`) | nobody | program | none — template-required by `reduce`; allocated, never written or read; 2 KB |
| `cb_membership` | `K·Ng` | `K·Ng` | `{N: —, HW: —, C: spans → K, G: spans → Ng}` | `Float32` (both matmul inputs must be fp32 for `HiFi4 + fp32 DEST`) | writer | compute (aggregation matmul in1, `NoWaitNoPop`) | program | none — live for the whole program (reduce-side mask, every image) |
| `cb_membership_t` (verifier) | `K·Ng` | `K·Ng` | same | `Float32` | writer (same build loop, transposed element address) | compute (expansion matmul in1, `NoWaitNoPop`) | program | none — the apply-side mask. Holding Mᵀ costs `K·Ng` tiles but makes every matmul in the op the same non-transposed `(1×K')@(K'×N)` body: the compute binary must fit the kernel-config ring in the `--dev` build (a transposed instantiation measured 4.9 KB) |
| `cb_gamma_rows` (if gamma) | `K` | `K` | `{N: —, HW: —, C: spans → K, G: —}` | affine dtype (bf16 for bf8b weights) | reader (`fill_affine_rows`: ROW_MAJOR weight → stick slices; TILE weight → row 0 of tile `c/32`, lane-gathered per 16-lane face row, bf8b decoded lane by lane — Refinement 4) | compute | program | none — held for every image; 2–4 KB/column |
| `cb_beta_rows` (if beta) | `K` | `K` | same | affine dtype (bf16 for bf8b weights) | reader (as above) | compute | program | none — as above |
| `cb_colsum_rows` | `K` (`2K` in `two_pass` programs — Refinement 5) | `K` (`2K`: S and U) | `{N: streams, HW: — (collapsed) / streams (raw running sum between chunks), C: spans → K, G: —}` | `Float32` | compute (reduce) | compute (reduce reload; matmul in0; two_pass: the variance-combine chain) | pass 1, then pass 2 (sequential); two_pass: pass A + the combine | **reused across pass 1 and pass 2** (same shape/format, disjoint lifetimes); **also the cross-chunk `Accumulate` accumulator** (implemented: the planner's separate `cb_colsum_accum` was merged — each output's reload pops its tile before that output's pack re-fills the slot, so `K` pages hold both roles); not with `cb_mean_rows` — both live at the pass-2 boundary |
| `cb_partial_rows` | `Ng` | `Ng` | `{N: streams, HW: —, C: —, G: spans → Ng}` | `Float32` | compute (matmul) | writer (multicast source) | per round | none — consumed by a *dataflow* kernel, so it cannot alias any compute-consumed CB (ownership rule) |
| `cb_gather` | `Ng·GT` (single landing region) | `Ng·GT` per round | `{N: streams, HW: — , C: —, G: spans → Ng}` + non-block `GT` | `Float32` | peers' writers (64 B row unicasts into the **root**'s region, row-tile-major tile `rt·Ng + k` = the matmul's K×N layout; the root's writer pushes after its monotone round counter reaches `(image+1)·num_active`) | root compute (the shared matmul body: `(1×GT)@(GT×Ng)` against `cb_inv_rows`) | per image, per round | none — ONE region suffices under the root-reduce protocol: a core sends its round-`r` rows only after receiving the previous round's broadcast, which the root sends only after its matmul consumed the previous gather (pushed and popped `Ng·GT` per round, so the landing address is the base every round). Rows ≥ `num_active` of the last row-tile zeroed once by the root. Allocated on every core (landing address must be identical), only the root's is written |
| `cb_group_mean` | `Ng` | `Ng` | `{N: streams, HW: —, C: —, G: spans → Ng}` | `Float32` | **writer** (broadcast landing; the root's compute combines into `cb_stats_bcast`) | compute (expansion in0, retained across the `K` calls) | image round | not with `cb_group_var`: both live in pass 3 (`shift` needs mean, `scale` needs var) |
| `cb_group_var` | `Ng` | `Ng` | same | `Float32` | **writer** (broadcast landing) | compute (expansion in0) | image round | carries the group **variance** (`Σ(x−mean)²·(1/n)` straight out of the combine matmul); popped by the pass-3 finalize chain (compute is its sole consumer) |
| `cb_group_rstd` (verifier) | `Ng` | `Ng` | same | `Float32` | compute (`CopyTile → +eps → Rsqrt → Pack` over the `Ng` group tiles) | compute (expansion in0, retained across the `K` calls) | pass 3 | compute-owned because `cb_group_var` is writer-produced (an in-place transform would make compute a second producer). Finalizing on `Ng` tiles instead of the `K` expanded rows keeps the exact SFPU rsqrt cost at `Ng` tiles (measured ~1 µs per tile) |
| `cb_stats_bcast` | `Ng` | `Ng` | same | `Float32` | root compute (combine matmul) | root writer (multicast source) | per round | none — consumed by a dataflow kernel; allocated identically on every core (addresses must match for the multicast landing) |
| `cb_inv_rows` (verifier) | `GT` | `GT` | `{N: —, HW: —, C: —, G: —}` + non-block `GT` | `Float32` (row 0 = `1/(HW·Cg)` in every lane, rows 1..31 zero) | root writer (filled once; re-pushed per round — the matmul pops its in0 and the bytes persist in the ring) | root compute (combine matmul in0) | program | none — root-only in0; allocated on every core for identical CB layout (4 KB × GT) |
| `cb_mean_rows` | `K` | `K` | `{N: streams, HW: —, C: spans → K, G: —}` | `Float32` | compute (matmul) | compute (chain B operand) | pass 2; pass-3 setup | **reused**: pass-2 mean rows, then re-expanded mean rows for the shift computation in pass 3 (popped tile-by-tile as `shift` is produced); not with `cb_scale_rows` — both live during `shift_full_block` |
| `cb_scale_rows` | `K` | `K` | same | `Float32` | compute | compute | pass 3 | in place: expanded rstd rows × gamma → scale (FPU `mul`; no separate rstd-row CB) |
| `cb_shift_full` | `K` | `K` | same | `Float32` (full 32-row tiles) | compute (`unary_bcast<Row>`) | compute (pass-3 chain DestReuse operand) | pass 3 | could alias `cb_colsum_accum` + `cb_colsum_rows` (both idle in pass 3, together `2K` pages ≥ `K`): **not done in Phase 0** — different page counts per CB make the alias a two-CB union; recorded as the L1 lever to pull first if the resident budget is short |
| `cb_fp32_scratch` | `Q·K` | `Q·K` | `{N: streams, HW: streams → Q, C: spans → K, G: —}` | `Float32` | compute (pass-1 copy chain / pass-2 centered-square chain) | compute (accumulate reduce) | per pass-1 and pass-2 chunk | pass 3 does not use it (fused apply chain). Pass 1 stages through it so both passes share ONE accumulate-reduce instantiation (kernel-config ring) and the resident pass-1 overlaps the DRAM fill (cumulative wait per chunk) |
| `cb_output_tiles` | `D·Q·K` | `Q·K` | `{N: streams, HW: streams → Q, C: spans → K, G: —}` | output dtype | compute | writer (TILE) / compute untilize (RM) | per chunk | none — consumed by a dataflow kernel; `D` for compute/writer overlap |
| `cb_output_sticks` (RM only) | `D·K` | `K` | `{N: streams, HW: streams → 1 tile-row, C: spans → K, G: —}` | output dtype | compute (untilize) | writer | per tile-row | none — dataflow consumer; `D` for overlap |
| `cb_input_tiles` **(TILE block shard, Refinement 1)** | the shard: `Hs·K` tiles, **placed on the input shard buffer** (`ttnn.cb_descriptor_from_sharded_tensor`, zero-copy — no allocation, no NoC read of the local shard) | `Hs·K` (image n's sub-block at offset `row_off·K`; all three passes read it in place) | `{N: spans → the images the shard straddles, HW: spans → Hs, C: spans → K, G: —}` | input dtype | reader (credits only: `Hs·K` per image with work) | compute | program (the shard) | shares its L1 with the **output** when `in_place` (second buffer index on the same region — pass 3 reads each input tile for the last time exactly once, in the block the pack lands in) |
| `cb_output_tiles` **(TILE block shard)** | the output shard: `Hs·K` tiles, placed on the output shard buffer (or the input shard when `in_place`) | `Hs·K` | same | output dtype | compute (pack) | nobody (the buffer IS the result) | program | with `cb_input_tiles` when `in_place` (see above); no `D`-deep stream — the writer moves nothing |
| `cb_input_shard` / `cb_output_shard` **(ROW_MAJOR block shard)** | the shard as stick pages (`shard_rows` sticks of `shard_w·elem` B), placed on the shard buffer(s) | the shard | `{N: spans, HW: spans → shard_rows sticks, C: spans → shard_w lanes, G: —}` | input / output dtype | reader stages sticks OUT of `cb_input_shard` (L1 → L1, `K·64 B` stride, pad lanes / pad sticks zeroed); writer copies the valid `c_valid·elem` B of each untilized stick INTO `cb_output_shard` | — | program | one region when `in_place` (the tiled copy in `cb_input_tiles` is complete before pass 3 writes the shard); the RM shard cannot back `cb_input_sticks` directly because a 40-channel stick (80 B) is not the tilize's `K·64 B` stick |
| `cb_input_sticks` / `cb_output_sticks` **(ROW_MAJOR block shard, direct view — Refinement 3)** | the shard, **placed on the shard buffer(s)** and re-paged as tile-sized pages: `Hs'·K'` pages where `K' = lcm(shard_w, 32)/32`, `Hs' = shard_rows/(32·m)`, `m = lcm/shard_w` (a `[2048, 40]` bf16 shard is byte-identical to a `[512, 160]` row-major block: `K' = 5`, `Hs' = 16`) | the shard | `{N: spans → the images the shard covers, HW: spans → Hs' view rows (m sticks each), C: spans → K'·32 view lanes = m periods of shard_w channels, G: —}` | input / output dtype | reader (credits only: `rows·K'` per image with work — no NoC read of the local shard, no staging); compute's `untilize` packs straight into `cb_output_sticks` | compute `tilize` (front to back: image n's rows are consecutive and images are processed in order) / nobody (the output buffer IS the result) | program | one region when `in_place` (pass 1 tilizes image n's rows out of it before pass 3 untilizes over them); replaces `cb_input_shard`/`cb_output_shard` AND the allocated `D·K` stick CBs (`fixed_footprint(..., rm_direct=True)` drops the `2·TB·D·K` term). Taken iff the view is exact (`_rm_direct_view`: stick page = `shard_w·elem` B, `K' ≤ MAX_CORE_C_TILES`, `shard_rows` and `HW` multiples of `32·m`, so no view row is partly outside its image) AND the K'-proportional fixed CBs fit below the shard; otherwise the staged rows above apply. View lane `j` is channel `c0 + j % shard_w` — the membership build and the affine rows take `c_period` (host common arg; `= K·32`, the identity, on every other path) |
| `cb_inv32_row` (`two_pass` programs — Refinement 5) | 1 | 1 | constant | `Float32` (row 0 = 1/32 in every lane, rows 1..31 zero) | writer (once) | compute (in0 of the shift matmul `(1×1)@(1×K)` over chunk-0 tile-row 0, `WaitAndRetain`, never popped) | program | none — 4 KB constant; the shift s it produces is what lets pass A square centered values before the group mean exists |
| `cb_zero_row` (sharded) | 1 page of 128 B | 128 B | constant | `Float32` (zeros) | writer | writer (unicast source) | program | the partial row a core sends to the root for an image its shard has no sticks in, so the root's per-image row count is uniform (N > 1 shards) |
| `cb_masked_mean` (`hw_mask` programs only: a tile-row partly outside its image — RM shard heights that are stick counts, N > 1 straddles, and since Refinement 2 `HW % 32 != 0` in any placement / regime) | `2·Ng` | `Ng` per masked segment (head, tail) | `{N: streams, HW: — , C: —, G: spans → Ng}` | `Float32` | writer (copy of the landed group-mean tiles with the rows outside the image zeroed) | compute (expansion matmul in0 for that segment's pass-2 mean operand) | pass 2 | replaces the design's `cb_hw_mask`: the row mask enters through the SAME expansion matmul (`mean_full[r][c] = mask[r]·mean_c`), so pass 2 stays ONE chain (a mask multiply element measured +1.1–10 KB of --dev code); allocated iff `hw_mask`, counted in `fixed_footprint(..., hw_mask)` |

## Symbol table

| Symbol | Bound | Predicate that establishes it |
|--------|-------|-------------------------------|
| `K = block_c_tiles = core_c_tiles` | `≤ min(config.MAX_CORE_C_TILES, Ct)` | host split search: `c_splits ≥ ceil(Ct / max_core_c_tiles_effective)` where `max_core_c_tiles_effective` is the largest `c ≤ MAX_CORE_C_TILES` with `fixed_footprint(c) ≤ L1_CB_BUDGET_BYTES` |
| `Ng = num_group_tiles = ceil(G/32)` | `≤ config.MAX_GROUP_TILES` (4) | host raises `NotImplementedError` above the cap (regime `sparse_membership`) |
| `GT = gather_tiles = ceil(num_active/32)` | `≤ ceil(num_cores/32)` (2 on an 8×8 grid, 5 on a 130-core grid) | `num_active ≤ num_cores` by construction |
| `Q = chunk_hw_tiles` | default `max(1, CHUNK_TILES_TARGET // K)` so `Q·K ≤ max(K, CHUNK_TILES_TARGET)` (≤ 32 tiles: the "Chunk size" lamp measured 32 ≥ 16 on every SDXL shape — −7 % at K=12, noise elsewhere); the split search halves `Q` (≥ 1) when that makes a TILE assignment resident (`SPLIT_PREFER_RESIDENT`) | definition + residency search |
| `H = block_hw_tiles` | resident: `core_hw_tiles` (bounded by the resident predicate below); streaming: `Q` | regime predicate |
| `core_hw_tiles` | `≤ ceil(HWt / hw_splits)` | balanced split |
| `D = STREAM_DEPTH` | 2 | config |
| `TB`, `T4`, `TA` | 1088–4096 / 4096 / 2048–4096 | dtype, affine dtype (bf8b weights → bf16 rows) |

## Total per-core footprint (closed form)

```
fixed_footprint(K, Ng, GT, Q, layout, has_gamma, has_beta) =
      T4 · K · Ng                       # cb_membership   M        (scales with K, Ng)
    + T4 · K · Ng                       # cb_membership_t Mᵀ       (K, Ng)
    + 2048                              # cb_scaler                (constant)
    + TA · K · (has_gamma + has_beta)   # cb_gamma_rows, cb_beta_rows (K)
    + T4 · K                            # cb_colsum_rows           (K)
    + T4 · Ng                           # cb_partial_rows          (Ng)
    + T4 · Ng · GT                      # cb_gather (one landing region) (Ng, GT)
    + T4 · GT                           # cb_inv_rows              (GT)
    + T4 · 2 · Ng                       # cb_group_mean, cb_group_var (Ng)
    + T4 · Ng                           # cb_group_rstd            (Ng)
    + T4 · Ng                           # cb_stats_bcast (Ng) — root combine + broadcast
    + T4 · K                            # cb_mean_rows             (K)
    + T4 · K                            # cb_scale_rows            (K)
    + T4 · K                            # cb_shift_full            (K)
    + T4 · Q · K                        # cb_fp32_scratch          (Q·K ≤ 16 tiles)
    + [two_pass] T4 · K + T4            # cb_colsum_rows 2K (S and U accumulators) + cb_inv32_row (Refinement 5)
    + TB · D · Q · K                    # cb_output_tiles          (D, Q·K)
    + [RM] TB · D · K                   # cb_input_sticks          (D, K)
    + [RM] TB · D · K                   # cb_output_sticks         (D, K)

input_footprint = resident ? TB · core_hw_tiles · K : TB · D · Q · K

total = fixed_footprint + input_footprint  ≤  config.L1_CB_BUDGET_BYTES   (1_000_000 B at Phase 0)
```

Worked values (bf16, 110 cores, `Ng = 1`, `GT = 4`): `K = 10, Q = 3` (SD C = 320, RM, gamma+beta): fixed = 40 + 40 + 2 + 40 + 40 + 4 + 16 + 16 + 8 + 4 + 4 + 40 + 40 + 40 + 120 + 120 + 40 + 40 KB ≈ **654 KB** → `(1,1,16384,320)` RM at 5×10 tiles = 100 KB is resident. `K = 4, Q = 4` (VAE `(1,1,65536,512)` TILE): fixed ≈ 230 KB → input 304 tiles = 608 KB → resident (the split search shrank `Q` from 8 to 4 to get there). Terms that scale with `K` dominate; `MAX_CORE_C_TILES` is the lever (perf lamp), then the `cb_shift_full` alias noted above. The `fixed_footprint()` function in the program descriptor is the executable form of this closed form and is what the split search evaluates.

## Data-movement budget

Chosen split: 2-D `(hw_splits × c_splits)` tile split with the stats all-gather; `V = N·HW·C·elem_bytes`.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input `x` | **1** in `resident_2d` (whole per-core assignment held across all three passes); **2** in `streaming_2d` with `two_pass` (Refinement 5: pass A takes both statistics from one read of each chunk, pass B applies — measured 1.28–1.33× on the VAE cells, the 4V → 3V ratio); **3** on the `hw_mask` streaming path (HW % 32 ≠ 0: one read per pass) | residency decision = the regime predicate; `hw_splits` is what makes the slice fit | none for `x` itself |
| output `y` | 1 | written once per chunk | none |
| gamma, beta | `hw_splits` reads of each core's column slice (`≤ num_active · 2 · K · 64 B` total; 41 KB for C = 320 on 64 cores vs 10 MB of input) | not resident across cores; multicast not warranted at this size (operand-reuse check) | none |
| input `x` **(BLOCK_SHARDED, Refinement 1)** | **0** — the shard is the block and is consumed in its own L1 (TILE: the CBs are placed on the shard buffers; RM: sticks are re-staged L1 → L1 into the tilize CB, `shard_rows × shard_w·elem` B per core once per image) | `block_sharded_resident`: `(hw_splits, c_splits, per-core extents)` come from the shard spec, no split search | none for `x`; same stats traffic as interleaved, plus one 128 B zero row per core per image the shard misses (N > 1) |
| output `y` **(BLOCK_SHARDED)** | **0** — packed straight into the output shard (TILE) / valid sticks copied L1 → L1 into it (RM); `in_place=True` packs over the input shard | placement | none |
| group statistics | 0 | never touch DRAM | per image: 2 rounds × (`num_active × Ng × 128 B` unicast rows into the root + `num_active` atomics + one `Ng × 4 KB` multicast to the rectangle + `rect_cores` handshake atomics) — 14 KB payload and ~440 atomics at 110 cores, `Ng = 1` (was `num_active²` = 12 100 atomics per round with the all-gather) |

Totals per image (SDXL `(1,1,16384,320)` bf16, 110 cores, resident): DRAM ≈ `10.5 MB read + 10.5 MB written + ≤ 140 KB weights`; cross-core ≈ `28 KB payload + ~900 atomics`. Streaming would read 31.5 MB instead.

> Cheapest-traffic split considered: 2-D tile split with residency — input 1×, output 1×, ~28 KB cross-core per image. Implemented: the same split (`resident_2d`), with `streaming_2d` as the fits-in-L1 fallback (+2 input DRAM reads). **Verifier (Phase 0)**: the split search now ranks residency above the tile-count objective for TILE inputs and shrinks the chunk `Q` (down to 1 tile-row) when that is what makes the assignment fit — `(1,1,65536,512)` TILE moved from `K=16` streaming (735 µs) to `K=4, Q=4` resident (497 µs). ROW_MAJOR keeps the widest stick slice instead (same shape: `K=16` streaming 770 µs beat `K=4` resident 916 µs), so its residency comes only when free. See `config.py` for every measured knob. The implemented split is the cheapest; the `(n, g)` group split and the lcm-cluster split were rejected (duplicate boundary reads + conflicting output writes; ≤ 2-core parallelism), and `block_sharded_resident` (0 DRAM crossings) is the deferred placement of the same structure.

## Implementation notes (ttnn-implementer, Phase 0)

- **Uniform `K` across cores**: `tilize<K,…>` / `untilize<K,…>` take the block width as a template
  argument, so the split search restricts `c_splits` to divisors of `Ct` (every INPUTS shape has one
  inside the caps). `core_hw_tiles` still varies by at most one across core rows (balanced split).
- **CB quantum contract (ring-wrap safety)**: every streaming CB (`cb_input_tiles`, `cb_fp32_scratch`,
  `cb_output_tiles`) carries a *nominal* `Q·K` pages per chunk — a ragged tail chunk of `q < Q` rows still
  pushes `Q·K`, the last `(Q−q)·K` pages carry no data and are popped unread by the consumer. The resident
  block is padded to `Hmax·K` per image the same way. This keeps every linear block index (the reduce's
  indexed access, the writer's page loop) from straddling the ring boundary; capacities in the table are
  unchanged and the pads cost no DRAM traffic.
- **Combine data path (root-reduce + broadcast, the design's "Grid synchronization" lamp alternative)**: the
  flat all-to-all multicast all-gather measured ~5 µs per sender per round at 110 cores (1.15 ms on
  `(1,1,4096,320)`), so it was replaced: every active core unicasts its `2·Ng` 64 B partial rows into the
  root's `cb_gather` half + one atomic on the root's round counter; the root's compute reduces the `GT·Ng`
  gathered tiles into `cb_stats_bcast` (finalized: `1/n`, and `+eps`/`rsqrt` on round 1); the root's writer
  broadcasts the `Ng` stat tiles with `mcast_pipe` (`SenderPipe`/`ReceiverPipe`, one sender per round,
  pre-handshake) into every core's `cb_group_mean` / `cb_group_var`. Measured 34 µs on that shape.
- **Verifier (Phase 0) kernel-config-ring fixes**: the `--dev` (watcher) build of the RM `K = 10` cells
  overran the 70 656 B ring by 1.7 KB. Resolved without algorithmic change: the root combine is the
  aggregation matmul (no second `reduce` instantiation, no raw SFPU post-op), the expansions use a
  transposed membership copy so every matmul is ONE non-transposed body (`row_matmul_block`), and the
  grid-wide constants travel as common runtime args. Compute binary 57.2 KB → ~50 KB (dev), program
  size 72.3 KB → ~65 KB; L1 cost `+K·Ng·T4` (`Mᵀ`) `+ GT·T4` (`cb_inv_rows`) `+ Ng·T4` (`cb_group_rstd`)
  `− Ng·GT·T4` (the second gather half).
- **Pass 1 is chunked in both regimes and staged through `cb_fp32_scratch`** (copy chain, then the same
  accumulate reduce as pass 2): one reduce instantiation per binary instead of two (the program must fit
  the kernel-config ring under the `--dev` watcher build), and the resident regime's pass 1 now overlaps
  the DRAM fill (cumulative `cb_wait_front` per chunk — the "Overlap (resident pass 1)" lamp). Cost: one
  extra unpack/pack per input tile in pass 1; `cb_fp32_scratch` is live in passes 1 and 2 (same `Q·K`).
- **Chain block size** `b = largest divisor of K ≤ DEST_AUTO_LIMIT` (compile-time from `K`), so every
  `PerBlockSize` push/pop quantum divides the `K`-multiples the CBs are sized in.

## Implementation notes (Refinement 1 — block-sharded placement + in_place)

- **The shard is the block** (regime `block_sharded_resident`): `K = ceil(shard_w/32)`, `Hs = ceil(shard_rows/32)`,
  grid = the shard grid; `Q = CHUNK_TILES_TARGET // K` shrunk only if the fixed CBs (+ the RM tiled copy) do not
  fit next to the shard. Every core carries `(c0, c_valid, s0, sticks_valid)`; the three kernels intersect the
  stick range with each image (`kernels/groupnorm_sc_N_1_HW_C_geometry.hpp`), so N > 1 shards — including shards
  that straddle two images mid tile-row — need no per-image runtime-arg tables.
- **TILE shards** are zero-copy on both sides: the reader pushes only credits; block offsets are shard-absolute
  (`row_off·K`); the pack lands at the same offsets, so `in_place` is a second buffer index on the input shard's
  CB region. **RM shards** cannot be consumed by the tilize directly (a `shard_w·elem` stick ≠ the `K·64 B` tilize
  stick), so the reader stages each tile-row's valid sticks L1 → L1 (pad lanes / pad sticks zeroed) and the writer
  copies the valid bytes of each untilized stick back; the tiled copy (`Hs·K·TB`) is the resident block.
- **Partial tile-rows** (RM shard heights not multiples of 32, N > 1 straddles): pass 1 is exact with zero pad
  sticks; pass 2 gets its mask through the expansion matmul (`cb_masked_mean`, above) — never through a chain element.
  `cb_inv_rows` carries `1/n` in all 32 rows in those (`hw_mask`) programs so the stat tiles are full tiles.
- **Stats flow control on N > 1 shards**: every active core reserves/pushes the landing CBs for EVERY image (the
  reserve is what keeps the next broadcast off stats a core is still reading in pass 3); compute drains them for
  images its shard misses. Cores without work in an image unicast a zero row so the root's count is uniform.
- **Kernel-config ring** (dev build): every sharded config fits after (a) one pass-2 chain per binary (mask via
  matmul, not a chain element: +10 KB / +1.1 KB measured for the two chain variants), (b) `pass2_segment` /
  `root_combine_block` as `noinline` bodies, (c) pops routed through the `noinline` `pad_pop`. The tightest cell
  (RM auto shard K = 2 with masks) sits ~1 KB under 70 656 B.

## Implementation notes (Refinement 2 — non-tile-aligned HW and C)

- **No new CB, no new helper instantiation.** `c_non_aligned` rides Refinement 1's per-core `(c0, c_valid)`
  valid-lane pair (host: `c_valid = min(K·32, C − c0)` for the interleaved leg too): membership rows / columns
  and gamma/beta lanes `≥ c_valid` are zero, so channels `≥ C` drop out of every group sum and the expansions
  yield zero output lanes; the ROW_MAJOR reader reads `c_valid·elem` B of each stick (`read_sticks_for_tilize`
  pads the L1 stride to `K·64 B`) into `cb_input_sticks` whose pad lanes were zeroed ONCE (nothing else writes
  them), and the writer writes `c_valid·elem` B per stick (`write_sticks_after_untilize`). `hw_non_aligned`
  rides the `hw_mask` masked-mean path: the host sets `hw_mask = HW % 32 != 0` for interleaved programs (and
  `_has_partial_rows` now sees TILE shards through the padded image stride `HWt·32`), the core owning the
  image's last tile-row gets `row_hi = HW − 32·(HWt − 1)` and the shared geometry hands every kernel the same
  `[0, hi)` tail; `cb_masked_mean` is allocated iff `hw_mask` (`+2·Ng·T4`, ≤ 32 KB).
- **Streaming regime + ragged tail**: compute's pass-2 `[head][body Q-chunks][tail]` segments are now the shared
  `pass2_segments` (geometry.hpp) and the streaming TILE reader pushes its nominal `Q·K` chunks in that order,
  so a 1-row tail chunk is never swallowed by a body chunk (RM streaming pushes per tile-row and needs no
  agreement). The chain body is unchanged; only the `x` operand policy follows the regime.
- **Aligned page sizes** (`buffer_aligned_page_size()`) travel as common runtime args for both tensors — a
  `C = 17` RM stick is a 64 B page, `C = 50` 128 B — replacing the `Ct·64` literal that coincided with them only
  at 64 B DRAM alignment.
- **Kernel-config ring**: unchanged instantiation count; every RM sharded / non-aligned config passes in `--dev`.

## Implementation notes (Refinement 3 — SDXL sharded ROW_MAJOR perf)

- **RM direct view** (`config.RM_SHARD_DIRECT_VIEW`): the RM shard is consumed in place as the row-major block of
  width `lcm(shard_w, 32)` (rows above). Per-core tiles on `(1,1,16384,320)` `[2048,40]`: 64 × 2 = 128 (37.5 % pad
  lanes) → 16 × 5 = 80, and the 2 × 2048 per-stick 80 B L1 → L1 NoC copies per core are gone. The resident tiled copy
  (`cb_input_tiles`, `Hs'·K'·TB`) is exactly the shard's byte size. Measured (8×8 model grid, in_place, device µs):
  16384×320 181.5 → 70.5, 4096×640 68.2 → 47.4, 16384×640 214.6 → 111.4, 1024×1280 38.7 → 33.8, 4096×1920 123.6 → 121.6.
- **Honest sharded CB ceiling**: the shards are allocated top-down in the same L1, so the sharded fit loop now uses
  `min(L1_CB_BUDGET_BYTES, lowest shard address − allocator L1 base − L1_CB_SAFETY_MARGIN_BYTES)`. `(1,1,16384,960)`
  `[2048,120]` (staged: 972 928 B of CBs vs a 969 728 B ceiling) clashed with the shard before; it now shrinks `Q` 8 → 4
  and runs (245 µs). The direct view (`K' = 15`) is skipped there — its K-proportional CBs (membership ×2, four K-row
  CBs, scratch) miss the ceiling even at `Q = 1` — and the staged view is the fallback; out-of-place shards of that
  size still cannot fit a resident tiled copy next to two shards (pre-existing).
- **Chain block size** `b = min(K, DEST_AUTO_LIMIT)` (was the largest divisor of `K ≤ 4`, i.e. 1 for `K ∈ {1, 5, 7, …}`):
  the chain blocks within a K-tile row with a valid-remainder tail, and every chunked CB is a whole number of rows, so
  no block can straddle the ring for any `b`. `K = 5` runs 4 + 1 tiles per DEST window instead of 1 + 1 + 1 + 1 + 1
  (16384×320: 72.6 → 70.5 µs). No CB size changes.

## Audit notes

1. **Capacity vs live set**: every gap is a stated mechanism — `D` double-buffering on the four streaming CBs, the two-round halves of `cb_gather`. No CB is sized to a block while its live set is one tile.
2. **Page format vs DEST width**: `fp32_dest_acc_en = True`; every compute-produced CB is `Float32`. The 16-bit exceptions are consumed-only (`cb_input_*`, `cb_gamma/beta_rows`, `cb_scaler`) or the tensor-dtype output.
3. **Disjoint lifetimes**: `cb_colsum_rows` (pass 1 ↔ pass 2) and `cb_mean_rows` (pass 2 ↔ pass-3 setup) are reused; `cb_scale_rows` is an in-place transform (rstd rows → scale); `cb_group_rstd` (pass 3 only) could share with `cb_partial_rows` (idle in pass 3, same `Ng` pages) but that CB is dataflow-consumed (ownership rule) — not aliased; the `cb_shift_full` ↔ `cb_colsum_*` alias is recorded as the first lever, not taken. `cb_membership_t`, `cb_inv_rows` are program-lifetime constants traded deliberately for compute code size (kernel-config ring).
4. **Bounds and closed form**: every non-block symbol (`K, Ng, GT, Q, D`) is bounded above with its predicate; the total is closed-form; the split search consults it before residency is decided (inventory first — the only "solve" is the monotone `c_splits` increase, and the buffer count is minimal: 15 CBs on the TILE leg, of which 4 are `Ng`-sized rows and 1 is a constant).
