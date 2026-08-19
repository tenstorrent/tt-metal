# GNA fused neighborhood-attention kernel — design spec

Goal: 3-D neighborhood attention at plain-flash utilization (~50% of peak vs the current windowed kernel's
~4%), by **reading each K/V tile from DRAM once and reusing it across adjacent query-blocks** (FlashAttention-
style tiling) and applying a **tile-aligned block mask** instead of a per-element mask. Target: 6s decode
fused-sdpa ~4.88s -> ~0.2-0.5s, decode ~19.5s -> ~15-16s. Exact (the box always covers each query's window).

## Why the existing ops can't do this (measured, this session)
- Current fused kernel: gather box per q-chunk (re-reads shared halo ~23x) + `use_windowed_narrowing` +
  per-element mask -> compute path ~4.6% util. Disabling windowed-narrowing walks the FULL K (k_num_chunks =
  padded_Sk/k_chunk), not the box -> not a fix.
- Plain SDPA over a materialized box: 52% util BUT the box must be materialized (embedding-gather = 23x DRAM,
  gather-bound) and the dense [vol,box] mask overflows L1 at box=8580.
- sparse_sdpa: per-QUERY-token index gather -> no cross-query halo reuse; index tensor ~12GB at 6s S.
- tile-skip on the fused path: blocked by the streaming subblock_h=1 wall.

## The validated foundation (already committed: models/tt_dit/layers/gna_gather.py)
- Block token permutation (block_permute.py) makes a query-chunk = one (bt,bh,bw) block, window compact.
- `mask_table(grid, block, kernel)` -> box_idx [nb,box], table [n_distinct<=27, vol, box] (TILE-quantizable),
  mask_id [nb]. Exact vs neighborhood (torch 100%, device op 99.975%). This is the kernel's geometry spec.

## Kernel structure (FlashAttention-neighborhood, K/V-reuse)
Process the sequence in **spatial groups** of adjacent query-blocks (a "super-block" region). Per region:

```
for region in regions:                          # region = a tile of adjacent query-blocks
    load K/V for region's BOX-UNION into L1 ONCE  # halo shared within the region is read once
    for qblk in region.query_blocks:            # each query-block reuses the resident K/V
        online-softmax flash(qblk.Q, resident K/V) with qblk's TILE-mask   # plain matmul path
        write qblk output
```

Key design decisions:
1. **Region size** trades L1 residency (K/V-union must fit L1) vs reuse (bigger region = each K/V tile reused
   by more query-blocks = closer to 1x DRAM). Tune to the (5,16,12)-class sweet spot (box/vol ~9, gather ~=
   compute).
2. **Tile-aligned mask**: precompute, per (query-tile, box-tile) pair, a single bit "any live pair?" from
   `mask_table` (quantize the [vol,box] mask to [vol/32, box/32]). Apply as a block-skip in the QK loop (skip
   fully-dead box-tiles per query-tile) + a residual per-element stamp ONLY on partial boundary tiles. This is
   the cheap mask that avoids the dense [vol,box] L1 blowup.
3. **K/V load = coalesced W-run gather** (reuse neighborhood_gather.hpp's wrow gather) into an L1 region CB,
   NOT re-gathered per query-block.
4. **Plain matmul path**: standard flash (sdpa.cpp compute primitives), NOT the windowed-narrowing path.
5. **W-SP**: the region's box-union may cross the W-shard edge -> the all-gathered (full-W) K/V feeds the
   gather (as today); Q stays W-sharded; per-device region set.

## Build phases
- P1 (this doc): spec + geometry (done/committed).
- P2: host program factory + op plumbing for the new op (mirror sdpa op; new reader/compute/writer).
- P3: reader = region K/V L1 load (coalesced) + Q-block streaming.
- P4: compute = per-query-block online-softmax flash over resident K/V + tile-mask block-skip.
- P5: writer + output scatter (block-order -> un-permute).
- P6: op parity gate (== exact neighborhood, 99.9%) + 6s decode timing; tune region size.

## Risks
- CB budget: region K/V-union + Q + scores + online-softmax state must fit L1 (drives region size).
- The tile-mask boundary-tile handling (partial tiles need a per-element stamp) — the one place per-element
  masking survives; keep it to boundary tiles only.
- W-SP region/offset bookkeeping (mirror the current per-device w_origin mechanism).
