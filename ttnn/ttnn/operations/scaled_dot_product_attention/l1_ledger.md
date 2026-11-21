# L1 Ledger: scaled_dot_product_attention

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking axes (from `op_design.md`): `batch`, `kv_head`, `group_member`, `q_tokens` (block extent `qc` = `q_chunk_tiles`), `k_tokens` (`kc` = `k_chunk_tiles`), `d_head` (`Dt` = `d_block_tiles`, pinned to the whole head). Plan `(R, C, W) = (4, 4, 2)`. Capacities are per cluster; STRIDED rings are `entries per NEO × C`, ALL rings `entries per producer region × R`. Formats follow the DEST width (`fp32_dest_acc_en`): Phase 0 has it on, so every compute-produced ring is Float32 (`T_acc` = 4,096 B); inputs/outputs bfloat16 (`T_in` = `T_out` = 2,048 B).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_q_block` | `C·Q_BUFFER_DEPTH·qc·Dt` | `C·qc·Dt` (this round) + up to `C·qc·Dt` (next round prefetched) | {batch: streams (one per round), kv_head: streams, group_member: streams (one member per NEO slice), q_tokens: spans → qc, k_tokens: streams (held across every chunk, independent of kc), d_head: spans → Dt} | input dtype (bfloat16) — dataflow payload into FPU srcB, never through DEST | reader | compute | whole round (held across all chunks) | none: live across every compute phase of the round; capacity = 2× live block is the `Q_BUFFER_DEPTH` double buffer (next round's Q lands during this one) |
| `cb_k_block` | `R·KV_BLOCKS_PER_READER·kc·Dt` | `kc·Dt` in use + up to `(R−1)·kc·Dt` prefetched | {batch: streams, kv_head: streams, group_member: streams (shared by all members via ALL), q_tokens: streams (shared by all q rows), k_tokens: spans → kc, d_head: spans → Dt} | input dtype (bfloat16) | reader | compute (ALL: every NEO) | one chunk (popped after `score_block`) | not with `cb_v_block`: V of chunk n is live while K of chunk n+1..n+R−1 is being prefetched into this ring (concurrent); capacity above the live block is the R producer regions of the ALL ring (one block per reader thread = prefetch depth R), a mechanism requirement of R producers on an ALL consumer |
| `cb_v_block` | `R·KV_BLOCKS_PER_READER·kc·Dt` | `kc·Dt` + up to `(R−1)·kc·Dt` prefetched | {batch: streams, kv_head: streams, group_member: streams, q_tokens: streams, k_tokens: spans → kc, d_head: spans → Dt} | input dtype (bfloat16) | reader | compute (ALL) | one chunk (popped after `pv_block`) | not with `cb_k_block` (concurrent, see above); not with `cb_scores`/`cb_probs` (different producer kernel and format; concurrent with both) |
| `cb_constants` | `CONST_TILES·C` (2 causal, 1 otherwise) | same — genuinely constant | {batch: streams (reused), kv_head: streams, group_member: streams, q_tokens: streams, k_tokens: streams, d_head: streams} — scales with none: a constant | `T_acc` format (Float32 when `fp32_dest_acc_en`; implementation change, see below) | reader | compute | whole kernel | already merged: scaler + diagonal mask share one ring. The design's third tile (full mask) is gone: key tiles entirely above the diagonal are filled with -inf in DEST (`fill_tile_bitcast`) and their matmul is skipped. Copy per NEO because a STRIDED ring gives each NEO only its own slice |
| `cb_scores` | `C·qc·kc` | `qc·kc` per NEO | {batch: streams, kv_head: streams, group_member: streams, q_tokens: spans → qc, k_tokens: spans → kc, d_head: streams (contracted inside `score_block`)} | Float32 (`fp32_dest_acc_en` on; follows the DEST width) | compute | compute | `score_block` → `exp_block` (one chunk) | not with `cb_probs`: both live during `exp_block` (S read while P written); an in-place pop/push rotation is impossible because the ring is indexed and must hold exactly the nominal block (straddle rule), so no free slot exists to push P0 before S is fully consumed. Not aliasable (`alias_with`) for the same overlap |
| `cb_probs` | `C·qc·kc` | `qc·kc` per NEO | {batch: streams, kv_head: streams, group_member: streams, q_tokens: spans → qc, k_tokens: spans → kc, d_head: streams} | Float32 | compute | compute | `exp_block` → `pv_block` | not with `cb_scores` (above); not with `cb_out_accum` (concurrent in `pv_block`) |
| `cb_row_max` | `C·qc` | `qc` per NEO (m_old popped before m_new is reserved, per row) | {batch: streams, kv_head: streams, group_member: streams, q_tokens: spans → qc, k_tokens: streams (one value per row across all chunks), d_head: streams} | Float32 | compute | compute | whole round | not with `cb_row_sum` / `cb_rescale`: all three live during `row_sum_block` (m held for the next chunk, α and l in use) |
| `cb_rescale` | `C·qc` | `qc` per NEO in the streamed regime; 0 in the resident regime (no chunk > 0), where the ring is kept because both regimes run the same kernels and bind the same rings: `C·qc·T_acc` = 16,384 B at `qc = 1`, accepted (verifier, Phase 0) | {batch: streams, kv_head: streams, group_member: streams, q_tokens: spans → qc, k_tokens: streams, d_head: streams} | Float32 | compute | compute | `row_max_block` → `rescale_accum_block` of the same chunk (chunk > 0 only) | could alias `cb_row_sum`'s lifetime? no: l is live at the same time (l = α·l + rowsum needs both). Could it be packed into `cb_row_max` as a second entry per row? no: m_new must be held (wait, no pop) through `exp_block` while α is popped earlier by `rescale_accum_block`; FIFO order of one ring cannot release α before m |
| `cb_row_sum` | `C·qc` | `qc` per NEO | {batch: streams, kv_head: streams, group_member: streams, q_tokens: spans → qc, k_tokens: streams, d_head: streams} | Float32 | compute | compute | whole round (l, then 1/l in place at round end) | the reciprocal is computed in place (no extra ring); not shared with `cb_row_max`/`cb_rescale` (concurrent) |
| `cb_out_accum` | `C·qc·Dt` | `qc·Dt` per NEO (rotated in place, pop before reserve per DEST section) | {batch: streams, kv_head: streams, group_member: streams, q_tokens: spans → qc, k_tokens: streams (accumulated over chunks), d_head: spans → Dt} | Float32 | compute | compute | whole round | rescale (`O ·= α`) is in place, P·V reloads and re-packs in place: one ring for O across all chunks. Not with `cb_out`: concurrent during `normalize_block`; different format (Float32 vs output dtype) |
| `cb_out` | `C·OUT_BUFFER_DEPTH` | 1 tile per NEO in flight + 1 being drained | {batch: streams, kv_head: streams, group_member: streams, q_tokens: streams, k_tokens: streams, d_head: streams} | output dtype (bfloat16) — the result ABI, packed from 32-bit DEST by the packer | compute | writer | per round (`normalize_block` → writer) | normalize packs straight into the writer's ring (pack-into-destination); capacity 2 per NEO is the `OUT_BUFFER_DEPTH` double buffer (compute packs tile k+1 while the writer drains tile k) |

Not a ring (decided, recorded here so the inventory is complete): the scaled/masked scores are produced in DEST by `score_block` and packed once (no separate "raw scores" buffer); exp is fused into the subtract chain (no "shifted scores" buffer); the normalization packs into `cb_out` (no "normalized output" buffer).

## Symbol table

| Symbol | Meaning | Bound | Predicate establishing it |
|--------|---------|-------|---------------------------|
| `R, C, W` | thread plan | `C ∈ {1,2,4}`, `R + W ≤ 6`, R and W each divide C, `C ≤ 4` (ALL fan-out) | `THREAD_PLAN` validated on the host (as `toy_row_passes.validate_thread_plan`) |
| `Q_BUFFER_DEPTH` | Q blocks per NEO | 2 | host constant |
| `KV_BLOCKS_PER_READER` | K/V blocks per producer region | 1 | host constant |
| `OUT_BUFFER_DEPTH` | output tiles per NEO | 2 | host constant |
| `CONST_TILES` | constant tiles per NEO | 2 (causal) / 1 | `is_causal` (host `const_tiles()`) |
| `qc` = `q_chunk_tiles` | query tile-rows per NEO unit | `1 ≤ qc ≤ Sqt`, and the ring caps and budget below | fit predicate (`op_design.md`, Work Distribution) |
| `kc` = `k_chunk_tiles` | key tile-rows per chunk | `1 ≤ kc ≤ min(Skt, kc_fit(qc))` | fit predicate |
| `Dt` = `d_block_tiles` | head-dim tiles | `C·Q_BUFFER_DEPTH·qc·Dt·T_in ≤ DFB_MAX_RING_BYTES`, `C·qc·Dt·T_acc ≤ DFB_MAX_RING_BYTES`, `R·Dt·T_in ≤ DFB_MAX_RING_BYTES` (at kc = 1) | fit predicate; failure → `NotImplementedError` (deferred `wide_head_dim`) |
| `T_in`, `T_out` | input / output tile bytes | 2,048 (bfloat16), 4,096 (float32) | dtype (`input_tensor_q.buffer_page_size()`) |
| `T_acc` | internal ring tile bytes | 4,096 when `fp32_dest_acc_en`, else 2,048 | resolved compute config |
| `DFB_MAX_RING_BYTES` | one ring's compute-side limit | 1,048,560 B | target page *Cluster model* (part limit, no run-time query) |
| `budget` | per-cluster L1 for rings | `ttnn.get_max_worker_l1_unreserved_size() − L1_RESERVE_BYTES`; **3,963,328 B** on quasar (probed), `L1_RESERVE_BYTES = 0` (the query already excludes the reserved region; the knob exists so a later refinement can hold back slack without a literal) | device query at run time |
| `Sqt = Skt`, `Dt` | `ceil(S/32)`, `ceil(D/32)` | any; only through `qc`, `kc`, `Dt` above | — |

## Total footprint (per cluster)

```
L1(qc, kc) =   C·Q_BUFFER_DEPTH·qc·Dt·T_in                 # cb_q_block      — scales with qc, Dt
             + 2·R·KV_BLOCKS_PER_READER·kc·Dt·T_in          # cb_k_block, cb_v_block — scales with kc, Dt, R
             + C·CONST_TILES·T_acc                          # cb_constants    — constant (CONST_TILES = 2 causal, 1 otherwise)
             + 2·C·qc·kc·T_acc                              # cb_scores, cb_probs — scales with qc·kc
             + 3·C·qc·T_acc                                 # cb_row_max, cb_rescale, cb_row_sum — scales with qc
             + C·qc·Dt·T_acc                                # cb_out_accum    — scales with qc, Dt
             + C·OUT_BUFFER_DEPTH·T_out                     # cb_out          — constant
           ≤ budget
```

Affine in `kc` for fixed `qc` (`fixed(qc) + kc·per_kc(qc)`), and every term is non-decreasing in `kc`. The
host evaluates exactly this expression and nothing else: `ring_entries()` in
`scaled_dot_product_attention_program_artifacts.py` is the one statement of every ring's capacity, and the
`DataflowBufferSpec`s, `SdpaPlan.l1_bytes()` and the fit predicate (`_fits` / `_kc_fit`: every ring
≤ `DFB_MAX_RING_BYTES`, the sum ≤ `budget`, coarsest `kc ≤ Skt` that holds) all read it. (Verifier, Phase 0:
the fit used to restate the capacities in a separate closed form; it now evaluates the ring table, checked
identical on 322,560 shape × plan × dtype × override cases.)

Worked on quasar, Phase 0 knobs (`qc = 1`, `kc = Skt`, `Dt` whole), plan (4, 4, 2), bfloat16 inputs, Float32
internal rings, from `ring_entries()` (verifier, Phase 0; replaces the planner's table, which predated the
Float32 constants ring and counted one K/V ring where there are two). Terms in the order of the footprint
expression: Q, K+V, constants, S+P, the three state rings, O, out. Full / causal differ only in the constants
ring (16,384 / 32,768 B).

| INPUTS entry | Skt, Dt | L1 (B), full | causal | of 3,963,328 (causal) | largest ring (B) ≤ 1,048,560 |
|--------------|---------|--------------|--------|-----------------------|------------------------------|
| (1,1,32,32) | 1, 1 | 16,384 + 16,384 + 16,384 + 32,768 + 49,152 + 16,384 + 16,384 = 163,840 | 180,224 | 4.5 % | 32,768 |
| (1,1,128,64) | 4, 2 | 32,768 + 131,072 + 16,384 + 131,072 + 49,152 + 32,768 + 16,384 = 409,600 | 425,984 | 10.7 % | 65,536 |
| (1,2,64,128) | 2, 4 | 65,536 + 131,072 + 16,384 + 65,536 + 49,152 + 65,536 + 16,384 = 409,600 | 425,984 | 10.7 % | 65,536 |
| (1,4,64,64) | 2, 2 | 32,768 + 65,536 + 16,384 + 65,536 + 49,152 + 32,768 + 16,384 = 278,528 | 294,912 | 7.4 % | 32,768 |
| (1,8,96,64) | 3, 2 | 32,768 + 98,304 + 16,384 + 98,304 + 49,152 + 32,768 + 16,384 = 344,064 | 360,448 | 9.1 % | 49,152 |
| (1,2,64,96) | 2, 3 | 49,152 + 98,304 + 16,384 + 65,536 + 49,152 + 49,152 + 16,384 = 344,064 | 360,448 | 9.1 % | 49,152 |
| (1,2,512,64) | 16, 2 | 32,768 + 524,288 + 16,384 + 524,288 + 49,152 + 32,768 + 16,384 = 1,196,032 | 1,212,416 | 30.6 % | 262,144 |

Outside the first phase, same knobs: `(1,2,200,64)` 606,208 / 622,592 B; `(2,2,100,64)` 409,600 / 425,984 B;
`(2,1,32,4096)` (Dt = 128) is refused by the fit at plan (4, 4, 2): `out_accum` is `C·qc·Dt·T_acc` =
4·1·128·4,096 = 2,097,152 B and `query_block` 4·2·1·128·2,048 = 2,097,152 B, both over the ring limit at
`qc = kc = 1`. At plan (1, 1, 1) it fits (1,601,536 / 1,605,632 B, largest ring 524,288 B) and passes (verifier
probe, PCC 0.999998): see op_requirements.md, Refinement 1.

## Data-movement budget

Chosen split: NEO units (group member × q chunk) of one kv head dealt C = 4 per cluster round; K/V shared by the round's NEOs through ALL rings; clusters independent. Counts in tiles (bfloat16, 2,048 B), Phase 0 knobs.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| Q | 1 (+ the padded NEOs' re-reads of NEO 0's unit: ≤ (C−1)·qc·Dt tiles per short round) | each unit's Q block is loaded once and held for all its chunks (`cb_q_block` resident across the round) | none |
| K | `rounds_per_head` per kv head (causal: each round reads only key rows up to its last query tile) | K/V of a head are re-read by every round of that head, because rounds run on different clusters and nothing holds K/V across rounds | none (sharing within a cluster is L1-local, ALL fan-out) |
| V | as K | as K | none |
| O | 1 (+ identical rewrites by padded NEOs / clamped rows) | normalize packs straight into the writer ring | none |

Per PHASE0 shape (non-causal / causal where they differ; Q and O include padding traffic):

| Shape | rounds (`rounds_per_head` × heads) | Q | K | V | O | K/V minimum |
|-------|------------------------------------|---|---|---|---|-------------|
| (1,1,128,64) | 1 | 8 | 8 / 8 | 8 / 8 | 8 | 8 |
| (1,2,64,128) | 1 | 16 | 8 / 8 | 8 / 8 | 16 | 8 |
| (1,1,32,32) | 1 | 1 + 3 | 1 | 1 | 1 + 3 | 1 |
| (1,4,64,64) | 1 × 4 | 16 + 16 | 16 / 16 | 16 / 16 | 16 + 16 | 16 |
| (1,8,96,64) | 3 × 2 | 48 | 36 / 24 | 36 / 24 | 48 | 12 |
| (1,2,64,96) | 1 × 2 | 12 + 12 | 12 / 12 | 12 / 12 | 12 + 12 | 12 |
| (1,2,512,64) | 8 × 1 | 64 | 256 / 144 | 256 / 144 | 64 | 32 |

Totals per tier for the largest PHASE0 shape `(1,2,512,64)`, non-causal: DRAM 640 tiles = 1,310,720 B (Q 131,072, K 524,288, V 524,288, O 131,072); cross-core NoC 0 B; core-local: K/V fanned out to 4 NEOs from one L1 copy (ALL ring), S/P/O/state never leave the cluster.

> Cheapest-traffic split considered: the chosen split plus read-once/multicast of each K/V chunk to every cluster of its kv head (`kv_multicast_across_clusters`) — DRAM K+V 1,048,576 B → 131,072 B on `(1,2,512,64)` (−917,504 B), at the cost of 7 × 131,072 B of NoC multicast payload and one semaphore handshake per chunk. Implemented: q-units of one kv head C per cluster with K/V shared within the cluster (4× below private per-NEO streams). Deferred because the built ALL ring is exactly the multicast's receive side (only the reader's producer side changes, so it layers on without restructuring), no Python `ProgramSpec` reference on this part exercises a cross-cluster multicast yet (`quasar_op_examples` README, "Not covered"), and its gain is bounded by injector serialization (catalog `shared_input_reuse`: 1.71× for an 11× read cut). The same reduction is reached per cluster by `kv_resident_across_rounds` whenever a cluster gets several rounds of one head.

## Implementation changes (ttnn-implementer, Phase 0)

- `cb_constants` is in the accumulator format (`T_acc`, Float32 here), not bfloat16. The row-max reduce moves
  DEST into SrcB, whose format is the scaler's.
- `cb_constants` holds 2 tiles when causal (scaler, diagonal mask), not 3: fully masked key tiles are filled
  with -inf in DEST and their matmul is skipped. Net per cluster (causal): 2·4·4,096 = 32,768 B vs the design's
  3·4·2,048 = 24,576 B (+8,192 B); non-causal: 16,384 B vs 8,192 B. Largest PHASE0 footprint `(1,2,512,64)`
  causal: 1,212,416 B of 3,963,328 (30.6 %).
- Every other ring, capacity and the data-movement budget are as planned.
