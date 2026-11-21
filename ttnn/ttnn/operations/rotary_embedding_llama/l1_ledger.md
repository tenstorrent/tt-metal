# L1 Ledger: rotary_embedding_llama

**Scope.** Sized for `quasar` (L1 4,194,304 B per cluster). Block axes are from `op_design.md`:

- `user` (extent `users_per_core` = 1);
- `head_row` (`head_row_tiles` = 1);
- `head_dim_tile` (`tensor_dim_tiles` = Wt per core, `neo_dim_tiles` = Wt / C per NEO);
- `channel_k` (`contraction_tiles` = 1, inside one tile).

**Thread plan.** C = `compute_threads`, clamped so that C divides Wt. STRIDED capacities are per NEO × C.

**Borrowed rings.** A borrowed ring's capacity *is* the tensor's own shard, so it allocates **0 bytes** of its own.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_tiles` | `tensor_dim_tiles` (= `neo_dim_tiles × C`), borrowed from x's shard | Wt: the whole shard is resident before launch and read by both compute phases | `{user: spans → users_per_core(1), head_row: spans → head_row_tiles(1), head_dim_tile: spans → tensor_dim_tiles, channel_k: spans → 1 (inside the tile)}` | input dtype (bf16 in Phase 0) | reader (credits only) | compute | whole program | **is** the x shard (zero-copy). It owns no allocation, so there is nothing to share. |
| `cb_cos_tiles` | `tensor_dim_tiles`, borrowed from cos's shard | Wt | `{user: spans → 1, head_row: streams → row 0 broadcast (no extent), head_dim_tile: spans → tensor_dim_tiles, channel_k: spans → 1}` | input dtype | reader | compute | whole program | **is** the cos shard (zero-copy) |
| `cb_sin_tiles` | `tensor_dim_tiles`, borrowed from sin's shard | Wt | same as `cb_cos_tiles` | input dtype | reader | compute | whole program | **is** the sin shard (zero-copy) |
| `cb_trans_mat` | 1, borrowed from T's shard (constant: one 32×32 tile) | 1 | `{user: streams (shared by all users), head_row: streams, head_dim_tile: streams (reused for every j via ALL), channel_k: spans → 1}` | input dtype | reader | compute (ALL) | whole program | **is** the T shard (zero-copy) |
| `cb_rotated` | `neo_dim_tiles × C` (= Wt); `rotated_depth_blocks` = 1 block | `neo_dim_tiles` per NEO: the whole share is packed by `rotate_block` before `rotate_combine_block` reads it (sequential phases, one block) | `{user: spans → 1, head_row: spans → 1, head_dim_tile: spans → neo_dim_tiles per NEO, channel_k: spans → 1 (contracted already)}` | **Input dtype** (bf16 in Phase 0), as implemented. The design had Float32 under a 32-bit DEST (audit 2). Keeping every FPU operand in one dtype keeps one ALU format for the whole kernel. The cost is one bf16 rounding of `x @ T` before the multiply by sin; the golden suite still passes at the 0.9997 PCC entry. | compute | compute (same NEO) | `rotate_block` → `rotate_combine_block` | **Pack into the destination:** not possible. The FPU row-broadcast multiply needs `rotated` as an L1 operand (srcA), not a DEST one; the output ring would have to be read back while being produced (two roles on one ring). **In place over x:** not possible, because x is read again by the `x·cos` term after `rotated` exists. **Alias:** no other non-borrowed ring exists. **Fold into DEST:** perf lamp. Catalog `compute_fusion` measured the L1 round trip faster than DEST reuse for an FPU consumer. |
| `cb_output_tiles` | `tensor_dim_tiles`, borrowed from the output shard | Wt | `{user: spans → 1, head_row: spans → 1, head_dim_tile: spans → tensor_dim_tiles, channel_k: spans → 1}` | output dtype | compute | writer (credits only) | whole program | **is** the output shard (zero-copy; R9 places it). The packer writes the result straight into it. |

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `tensor_dim_tiles` (Wt) | 1 ≤ Wt ≤ 4 | validate(): `head_dim % 32 == 0` (X4), `head_dim <= 128` (X7); `Wt = head_dim // 32` |
| `head_row_tiles` | = 1 | validate(): `heads <= 32` (X6) |
| `users_per_core` | = 1 | validate(): `users == 1` (TARGET `users = [1]`) |
| `compute_threads` (C) | C ∈ {1, 2, 4}, C ≤ `THREAD_PLAN.compute`, C divides Wt | host clamp (`op_design.md`, Mechanism caps) |
| `neo_dim_tiles` | = Wt / C ∈ {1, 2, 3, 4} | exact division by the clamp |
| `tile_bytes(fmt)` | 2048 (bf16), 4096 (Float32) | 32×32 tile |
| per-core budget | `ttnn.get_max_worker_l1_unreserved_size() - l1_reserve_bytes` (run-time query; L1 = 4,194,304 B on quasar, unreserved value read at run time) | no predicate needed: the op's own allocation is ≤ 16 KiB (below) |

## Total per-core footprint (beyond the caller's tensors)

```
own_L1 = cb_rotated
       = neo_dim_tiles × C × tile_bytes(rotated_fmt)
       = tensor_dim_tiles × tile_bytes(rotated_fmt)       (rotated_fmt = input dtype, see the cb_rotated row)
       ≤ 4 × 2048 B = 8 KiB     (quasar, Wt = 4, bf16; 16 KiB once float32 inputs are supported)
```

Borrowed rings: 0 B. They are the four input shards and the output shard, which the caller (inputs) and R9 (output, allocated with x's spec) place.

| Term | Scales with |
|------|-------------|
| `cb_rotated` | `tensor_dim_tiles` (the block extent along head_dim_tile), `rotated_fmt` (= input dtype). Unchanged by C, because the per-NEO share × C is Wt. |

Audits:

1. **Capacity = live set for every ring.** Borrowed rings are the whole resident shard. `cb_rotated` holds one block per NEO, the whole share.
2. **Page format.** `cb_rotated` is in the input dtype. This departs from "follow DEST width" because of the Quasar ALU-format constraint in its row. The other rings are the tensors' own shards, in the tensor's format.
3. **Disjoint lifetimes.** There is no pair of non-borrowed rings.
4. **Bounds.** Every symbol is bounded above.

## Data-movement budget

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input_tensor` (x) | 0 | L1-resident shard consumed in place through a borrowed ring; unpacked twice core-locally (matmul, x·cos) | 0 |
| `cos_cache` | 0 | resident, borrowed ring, unpacked once (row broadcast) | 0 |
| `sin_cache` | 0 | resident, borrowed ring, unpacked once | 0 |
| `trans_mat` | 0 | resident, one copy per core, fanned to the NEOs by an ALL ring (unpacked once per x tile, core-locally) | 0 |
| `rotated` (intermediate) | 0 | core-local pack + unpack once per tile | 0 |
| output | 0 | the packer writes straight into the output shard (borrowed ring) | 0 |

Total per tier, per call:

- **DRAM:** 0 B.
- **Cross-core NoC:** 0 B.
- **Core-local L1 reads** (unpacks), at most:
  - x: 2·Wt tiles;
  - cos and sin: Wt each;
  - T: Wt;
  - rotated: Wt.
- **Core-local L1 writes** (packs), at most: rotated Wt, output Wt.

> Cheapest-traffic split considered: `head_dim_tile` dealt over the shard core's NEOs, in place (0 B DRAM, 0 B NoC). Implemented: the same split. It is the cheapest; the alternative of splitting `head_dim_tile` over cores would add 4·Wt tiles + one T tile per core over the NoC, and it is rejected in the regime table.
