# Operation Design: scaled_dot_product_attention (Flash Attention)

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (matmul → online-softmax → matmul) |
| Goal | Compute `softmax(Q·Kᵀ·scale + mask)·V` using the Flash Attention algorithm: tile over the sequence dimension, maintain running max / running sum / running output per Q-row, and **never materialize the S_q×S_kv score matrix**. |
| Math | `O[b,h,i,:] = Σ_j softmax_j( (Q[b,h,i,:]·K[b,h,j,:])·scale + mask[i,j] ) · V[b,h,j,:]` |
| Mode | Hybrid (Flash Attention online softmax — mathematically exact in fp32 accumulators) |
| References | Tri Dao, *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*; `ttnn/cpp/ttnn/kernel_lib/{matmul_block_helpers,reduce_helpers_compute,reduce_helpers_dataflow,eltwise_convenience,eltwise_chain,eltwise_math,eltwise_binary_sfpu,eltwise_scalar,eltwise_fill}.hpp`; `.claude/references/generic_op_template/`; `eval/golden_tests/scaled_dot_product_attention/{feature_spec,helpers}.py` |

### Load-bearing constraint

The S_q × S_kv attention matrix is **never** held whole — not in DRAM, not in any L1 CB. The only score buffer is one per-block tile group of size `B_q × B_kv` (`cb_scores`). Per Q-chunk, all KV-chunks are streamed once and folded into running statistics (`m_i`, `l_i`, `O_i`). L1 footprint is O(B_q·B_kv + B_q·D + B_kv·D) — independent of S_kv. This is what distinguishes the op from plain SDPA.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `query` | ttnn.Tensor | yes | (B, H_q, S_q, D), TILE, bf16 | — | tensor |
| `key` | ttnn.Tensor | yes | (B, H_kv, S_kv, D), TILE, bf16 | — | tensor |
| `value` | ttnn.Tensor | yes | (B, H_kv, S_kv, D), TILE, bf16 | — | tensor |
| `attn_mask` | ttnn.Tensor | no | (B, 1, S_q, S_kv) or (B, H_q, S_q, S_kv), additive | None | tensor (optional) |
| `is_causal` | bool | no | {False} Phase 0; {True} refinement | False | CT (`MASK_MODE`) |
| `scale` | float | no | finite > 0; None → 1/√D | None | CT (`SCALE` bits, host-resolved) |
| `compute_kernel_config` | ttnn.ComputeKernelConfig | no | exposes `fp32_dest_acc_en` | None (→ HiFi-default, fp32 DEST acc) | host → ComputeConfigDescriptor |

`is_causal` and `attn_mask` are mutually exclusive — passing both is a `ValueError`.
`scale` is resolved on host (`scale or 1/sqrt(D)`) and baked into the compute kernel as a compile-time float-bits arg; it is applied on-device by pre-scaling Q.

### Derived (host) constants

| Symbol | Meaning | Formula |
|--------|---------|---------|
| `d_t` | head-dim tiles | `D / 32` |
| `Sq_t` | Q seq tiles | `S_q / 32` |
| `Skv_t` | KV seq tiles | `S_kv / 32` |
| `Bq_t` | Q tiles per Q-chunk (compile-time tunable) | recommend 2 (=64 rows); ≤ `Sq_t` |
| `Bkv_t` | KV tiles per KV-chunk (compile-time tunable) | recommend 2 (=64 rows); ≤ `Skv_t` |
| `Nq` | Q-chunks | `ceil(Sq_t / Bq_t)` |
| `Nkv` | KV-chunks streamed per Q-chunk | `ceil(Skv_t / Bkv_t)` |
| `group` | GQA group size (Phase 0 = 1) | `H_q / H_kv` |

Phase 0 is tile-aligned, so all `ceil`s are exact. `Bq_t`/`Bkv_t` are tuned so that score-block, Q-block, K/V-block and O-block CBs all fit L1; for large `D` (`d_t` up to 32, D=1024) the implementer shrinks `Bq_t`/`Bkv_t` to 1. Output matmul subblocks must keep `out_subblock_h * out_subblock_w ≤ 4` under `fp32_dest_acc_en=True` (≤ 8 otherwise).

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | Q (B, H_q, S_q, D); K,V (B, H_kv, S_kv, D); mask (B,1,S_q,S_kv) or (B,H_q,S_q,S_kv) |
| Dtype | bfloat16 (Phase 0); float32 / bfloat8_b are refinements |
| Layout | TILE_LAYOUT (only — SDPA is tile-only by design) |
| Memory | DRAM or L1 interleaved |
| Alignment (Phase 0) | S_q, S_kv, D all divisible by 32 |

### Output

| Property | Value |
|----------|-------|
| Shape | (B, H_q, S_q, D) — same as Q |
| Dtype | bfloat16 |
| Layout | TILE_LAYOUT |
| Memory | DRAM interleaved (caller `memory_config` honored) |

## Support contract (op file — implementer declares)

`INPUT_TAGGERS` (op-local; imported by `test_golden.py`):

| Tagger | Output | Rule |
|--------|--------|------|
| `tag_alignment(inputs, axes)` | `tile_aligned` / `w_non_aligned` / `h_non_aligned` | Q's last two dims `(S_q, D)`: `w_non_aligned` if `D%32≠0`; else `h_non_aligned` if `S_q%32≠0`; else `tile_aligned`. |
| `tag_attention_kind(inputs, axes)` | `self` / `cross` | `self` if `inputs[0][-2] == inputs[1][-2]` (S_q==S_kv) else `cross`. |
| `tag_kv_heads(inputs, axes)` | `mha` / `gqa` / `mqa` | Compare `inputs[0][1]` (H_q) vs `inputs[1][1]` (H_kv): equal→`mha`; H_kv==1→`mqa`; else→`gqa`. |

`mask_mode` and `scale_mode` are **not** taggers — `validate()` derives them from scalar kwargs:
- `mask_mode`: `causal` if `is_causal`; `custom` if `attn_mask is not None`; else `none`. Raise `ValueError` if both `is_causal` and `attn_mask` set.
- `scale_mode`: `auto` if `scale is None` else `explicit`.
- `fp32_dest_acc_en`: read from `compute_kernel_config.fp32_dest_acc_en`; default `True` when config is `None`. **Honored, never silently forced** — `False` is accepted for bf16/bf8b and rejected for fp32 via EXCLUSION.

`SUPPORTED` (Phase 0):

| Axis | Phase 0 values | TARGET (refinement goal) |
|------|----------------|--------------------------|
| `dtype` | `[bfloat16]` | `+float32, +bfloat8_b` |
| `fp32_dest_acc_en` | `[True, False]` | same |
| `layout` | `[TILE_LAYOUT]` | same (TILE-only) |
| `alignment` | `[tile_aligned]` | `+w_non_aligned, +h_non_aligned` |
| `attention_kind` | `[self, cross]` | same |
| `kv_heads_mode` | `[mha]` | `+gqa, +mqa` |
| `mask_mode` | `[none, custom]` | `+causal` |
| `scale_mode` | `[auto, explicit]` | same |

`EXCLUSIONS` (Phase 0): `[]`. Armed by refinements:
- `{"dtype": float32, "fp32_dest_acc_en": False}` — arms when fp32 lands (maxed input + non-maxed acc is rejected).
- `{"mask_mode": "causal", "attention_kind": "cross"}` — arms when causal lands (causal requires S_q == S_kv).

Non-axis shape-contract checks in `validate()` / entry point (raise `ValueError`/`RuntimeError`): rank == 4 for all tensors; `Q.D == K.D`; `K.S_kv == V.S_kv`; `K.shape == V.shape`; `Q.B == K.B`; `H_q % H_kv == 0`; mask (if present) shape ∈ {(B,1,S_q,S_kv),(B,H_q,S_q,S_kv)}.

## Dataflow Strategy

```
DRAM                         Tensix (per work-unit = one (b, h, q-chunk))                       DRAM
─────                        ────────────────────────────────────────────────────             ─────
Q[b,h,qc]  ──reader──▶ cb_q_in ──scale(MulUnary)──▶ cb_q (HELD across kv loop)
                                                         │
   for j in 0..Nkv-1:                                    │
K[b,hk,j]  ──reader──▶ cb_k_in ───────────────────┐      │
                                          matmul(transpose)  ──▶ cb_scores  (B_q×B_kv ONLY)
mask[..,j] ──reader──▶ cb_mask_in ──add──▶ cb_scores (custom mode)
                                  rowmax ──▶ cb_m_cur ; BinaryMax(cb_m) ──▶ cb_m_new
                                  exp(cb_scores − cb_m_new, bcastCol) ──▶ cb_p
                                  rowsum(cb_p) ──▶ cb_l_cur
                                  corr = exp(cb_m − cb_m_new) ──▶ cb_corr
                                  cb_l ← corr·cb_l + cb_l_cur
V[b,hk,j]  ──reader──▶ cb_v_in ──┐ matmul(cb_p, cb_v) ──▶ cb_pv
                                  cb_o ← corr·cb_o + cb_pv     (running output, HELD)
                                  cb_m ← cb_m_new
   end for
                          recip(cb_l) ; cb_out = cb_o · (1/cb_l) (bcastCol) ──writer──▶ O[b,h,qc]
```

Within a Tensix: reader (NCRISC) streams tiles into input CBs; compute (3 TRISCs) runs the recurrence entirely through CBs; writer (BRISC) drains `cb_out`. No inter-Tensix communication — each work-unit is independent, no semaphores, no multicast.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one `(b, h, q-chunk)` — a `Bq_t`-tall slice of Q for one (batch, head). |
| Total units | `B * H_q * Nq` |
| Grid | `ttnn.split_work_to_cores(compute_grid, total_units)` → contiguous unit range per core |
| Per-core work | iterate its assigned units; for each, stream all `Nkv` KV-chunks once |
| KV-head map | `h_kv = h / group`. Phase 0 `group==1` → `h_kv = h`. (GQA/MQA refinement: same formula, `group>1`.) |
| Remainder | uneven `total_units / num_cores` split by `split_work_to_cores` (some cores get +1 unit); each core's runtime args carry `(start_unit, num_units)`. |

Tile-aligned Phase 0 ⇒ every chunk is full (`Bq_t`/`Bkv_t` tiles); the last chunk may be partial when `Sq_t % Bq_t ≠ 0` — handled by per-chunk tile counts in runtime args (still tile-granular, never sub-tile in Phase 0).

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_q_in` | 0 | tile(bf16) | `2·Bq_t·d_t` | bf16 | reader | compute (scale) | per work-unit, double-buffered |
| `cb_k_in` | 1 | tile(bf16) | `2·Bkv_t·d_t` | bf16 | reader | compute (QKᵀ) | per KV-chunk, double-buffered |
| `cb_v_in` | 2 | tile(bf16) | `2·Bkv_t·d_t` | bf16 | reader | compute (PV) | per KV-chunk, double-buffered |
| `cb_mask_in` | 3 | tile(bf16) | `2·Bq_t·Bkv_t` | bf16 | reader | compute (mask add) | custom mode only; double-buffered |
| `cb_scaler_max` | 8 | tile(bf16) | 1 | bf16 | reader (prep) | compute (rowmax) | resident whole kernel |
| `cb_scaler_sum` | 9 | tile(bf16) | 1 | bf16 | reader (prep) | compute (rowsum) | resident whole kernel |
| `cb_p` | 10 | tile(bf16) | `Bq_t·Bkv_t` | bf16 | compute (exp) | compute (rowsum, PV) | per KV-chunk; held between exp and PV |
| `cb_o` | 11 | tile(bf16) | `Bq_t·d_t` | bf16 | compute (init/J) | compute (J), writer-feed | **persistent across kv loop** |
| `cb_pv` | 12 | tile(bf16) | `Bq_t·d_t` | bf16 | compute (PV matmul) | compute (O update) | per KV-chunk |
| `cb_o_resc` | 13 | tile(bf16) | `Bq_t·d_t` | bf16 | compute (J1) | compute (J2) | per KV-chunk |
| `cb_recip_l` | 14 | tile(bf16) | `Bq_t` | bf16 | compute (finalize) | compute (finalize) | per work-unit |
| `cb_out` | 16 | tile(bf16) | `2·Bq_t·d_t` | bf16 | compute (finalize) | writer | per work-unit, double-buffered |
| `cb_q` | 24 | tile(bf16) | `Bq_t·d_t` | bf16 | compute (scale) | compute (QKᵀ) | **persistent across kv loop** (retained, popped at unit end) |
| `cb_scores` | 25 | tile(bf16) | `Bq_t·Bkv_t` | bf16 (fp32-acc via DEST) | compute (QKᵀ) | compute (mask/rowmax/exp) | per KV-chunk |
| `cb_m_cur` | 26 | tile(bf16) | `Bq_t` | bf16 | compute (rowmax) | compute (BinaryMax) | per KV-chunk |
| `cb_m` | 27 | tile(bf16) | `Bq_t` | bf16 | compute (init/K) | compute (BinaryMax, corr) | **persistent across kv loop** |
| `cb_m_new` | 28 | tile(bf16) | `Bq_t` | bf16 | compute (BinaryMax) | compute (exp, corr, K) | per KV-chunk |
| `cb_l` | 29 | tile(bf16) | `Bq_t` | bf16 | compute (init/H) | compute (H, finalize) | **persistent across kv loop** |
| `cb_l_cur` | 30 | tile(bf16) | `Bq_t` | bf16 | compute (rowsum) | compute (H) | per KV-chunk |
| `cb_corr` | 31 | tile(bf16) | `Bq_t` | bf16 | compute (corr) | compute (H, J1) | per KV-chunk |

All `Bq_t×1` running/temporary vectors store the row-reduction result in tile column 0 (REDUCE_ROW output convention). Persistent CBs (`cb_q`, `cb_o`, `cb_m`, `cb_l`) are sized to exactly one block (not double-buffered) and are read/written in place across the kv loop. Score, P, scaler, and all per-block intermediate CBs are sized to the **block**, never to S_q×S_kv — this is the O(S) memory guarantee in CB terms.

**CB sync (producer push == consumer wait), verified:**
- `cb_q_in`: reader pushes `Bq_t·d_t` once/unit; scale waits `Bq_t·d_t` once. ✓
- `cb_k_in`,`cb_v_in`: reader pushes `Bkv_t·d_t` per kv-chunk; QKᵀ/PV wait `Bkv_t·d_t` per kv-chunk. ✓
- `cb_mask_in`: reader pushes `Bq_t·Bkv_t` per kv-chunk; mask-add waits same (custom only). ✓
- `cb_scores`: QKᵀ pushes `Bq_t·Bkv_t`; rowmax waits same (no-pop), exp waits same (pops). ✓
- `cb_p`: exp pushes `Bq_t·Bkv_t`; rowsum waits same (no-pop), PV waits same (pops). ✓
- `cb_q`: pushed once (scale); each QKᵀ waits `Bq_t·d_t` (retain, no pop); popped once at unit end. ✓
- `cb_m`/`cb_l`/`cb_o`: init pushes once; each iteration's update op pops then re-pushes (in-place stream). ✓
- `cb_scaler_*`: reader prepares 1 tile each; reduce waits (no pop), resident. ✓

## API Mapping

| Phase | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Requirements |
|-------|------|----------|-----------|------------------------|----------|-----------|--------------|
| boot | raw_api | `compute_kernel_hw_startup` | (compute API) | `(cb_q_in, cb_k_in, cb_out)` | — | — | once at kernel top |
| boot | raw_api | `mm_block_init` | matmul_block_helpers.hpp (see InitMode doc L138) | once at boot before any `matmul_block` | — | — | helper's `InitMode::Short` handles per-call short-init thereafter |
| scaler prep (max) | helper | `prepare_reduce_scaler` | reduce_helpers_dataflow.hpp:65 | `<cb_scaler_max, PoolType::MAX, ReduceDim::REDUCE_ROW>(1.0f)` | — | `cb_scaler_max` | pool-type-aware overload; reader-side |
| scaler prep (sum) | helper | `prepare_reduce_scaler` | reduce_helpers_dataflow.hpp:65 | `<cb_scaler_sum, PoolType::SUM, ReduceDim::REDUCE_ROW>(1.0f)` | — | `cb_scaler_sum` | pool-type-aware overload; reader-side |
| init m/l/O | helper | `eltwise_chain` + `FillScalar` + `PackTile` | eltwise_chain.hpp:576; eltwise_fill.inl:13 | `FillScalar<D0>(-1e30f)`→`PackTile<cb_m>` ; `FillScalar(0)`→`cb_l` ; `FillScalar(0)`→`cb_o` | — | `cb_m`,`cb_l`,`cb_o` | fills running stats before kv loop |
| scale Q | helper | `unary` (`MulUnary`) | eltwise_convenience.hpp:126; eltwise_scalar.inl:56 | `unary<MulUnary<>, cb_q_in, cb_q>(Bq_t·d_t, MulUnary<>(scale_bits))` | `cb_q_in` | `cb_q` | once/unit; folds `scale` into Q |
| A. QKᵀ | helper | `matmul_block` | matmul_block_helpers.hpp:795 | `<transpose=true, …, in0_policy=WaitAndRetainOnLastBlock, in1_policy=WaitAndPopPerKBlock>`, `MatmulBlockShape::of(M=Bq_t, N=Bkv_t, sb_h, sb_w, in0_block_k=d_t, num_k_blocks=1)` | `cb_q`(in0, retained), `cb_k_in`(in1, popped) | `cb_scores` | transpose=true ⇒ Q·Kᵀ, reduction over D; num_k_blocks=1 ⇒ Q never popped |
| B. mask add (custom) | helper | `add` | eltwise_convenience.hpp:55 | `add<cb_scores, cb_mask_in, cb_scores, BroadcastDim::None>(Bq_t·Bkv_t)` | `cb_scores`,`cb_mask_in` | `cb_scores` (in place) | skipped in `none` mode |
| C. rowmax | helper | `reduce` | reduce_helpers_compute.hpp:400 | `<PoolType::MAX, ReduceDim::REDUCE_ROW, ReduceInputPolicy::WaitUpfrontNoPop>`, `ReduceInputBlockShape::of(Bq_t, Bkv_t)` | `cb_scores`(no pop), `cb_scaler_max` | `cb_m_cur` | must not pop `cb_scores` (reused by exp) |
| D. m_new=max(m,m_cur) | helper | `binary_sfpu` (`BinaryMax`) | eltwise_convenience.hpp:146; eltwise_binary_sfpu.hpp:71 | `binary_sfpu<BinaryMax<>, cb_m, cb_m_cur, cb_m_new, ALife=HeldStream>(Bq_t)` | `cb_m`(held), `cb_m_cur` | `cb_m_new` | `cb_m` retained (needed by corr); `cb_m_cur` popped |
| E. P=exp(S−m_new) | helper | `eltwise_chain` (`BinaryFpu` Sub + `Exp` + `PackTile`) | eltwise_chain.hpp:576; eltwise_math.inl:27 | `eltwise_chain(EltwiseShape::grid(Bq_t,Bkv_t), BinaryFpu<cb_scores,cb_m_new,Sub,BroadcastDim::Col, BLife=HeldStream>{}, Exp<>{}, PackTile<cb_p>{})` | `cb_scores`(pop), `cb_m_new`(held) | `cb_p` | fused sub+exp share one DST window; `cb_m_new` retained for corr |
| F. rowsum | helper | `reduce` | reduce_helpers_compute.hpp:400 | `<PoolType::SUM, ReduceDim::REDUCE_ROW, ReduceInputPolicy::WaitUpfrontNoPop>`, `ReduceInputBlockShape::of(Bq_t, Bkv_t)` | `cb_p`(no pop), `cb_scaler_sum` | `cb_l_cur` | must not pop `cb_p` (reused by PV) |
| G. corr=exp(m−m_new) | helper | `eltwise_chain` (`BinaryFpu` Sub + `Exp` + `PackTile`) | eltwise_chain.hpp:576 | `eltwise_chain(EltwiseShape::tiles(Bq_t), BinaryFpu<cb_m,cb_m_new,Sub,None, APolicy=Streaming, BPolicy=Streaming>{}, Exp<>{}, PackTile<cb_corr>{})` | `cb_m`(pop old), `cb_m_new`(pop) | `cb_corr` | consumes old `cb_m` and `cb_m_new` |
| H. l=corr·l+l_cur | helper | `mul` then `add` | eltwise_convenience.hpp:93,55 | `mul<cb_corr,cb_l,cb_l,None>(Bq_t)` (in place, corr held)… then `add<cb_l,cb_l_cur,cb_l,None>(Bq_t)` | `cb_corr`(held),`cb_l`,`cb_l_cur` | `cb_l` | corr held for J1; `cb_l` updated in place |
| I. PV | helper | `matmul_block` | matmul_block_helpers.hpp:795 | `<transpose=false, in0_policy=WaitAndPopPerKBlock, in1_policy=WaitAndPopPerKBlock>`, `MatmulBlockShape::of(M=Bq_t, N=d_t, sb_h, sb_w, in0_block_k=Bkv_t, num_k_blocks=1)` | `cb_p`(in0, pop), `cb_v_in`(in1, pop) | `cb_pv` | reduction over kv-chunk |
| J1. O_resc=corr·O | helper | `mul` | eltwise_convenience.hpp:93 | `mul<cb_o, cb_corr, cb_o_resc, BroadcastDim::Col>(Bq_t·d_t)` | `cb_o`(pop), `cb_corr`(pop) | `cb_o_resc` | corr `[Bq_t×1]` bcast across `d_t` cols |
| J2. O=O_resc+PV | helper | `add` | eltwise_convenience.hpp:55 | `add<cb_o_resc, cb_pv, cb_o, BroadcastDim::None>(Bq_t·d_t)` | `cb_o_resc`,`cb_pv` | `cb_o` | running output re-pushed |
| K. m=m_new | helper | `copy` | eltwise_convenience.hpp:171 | `copy<cb_m_new, cb_m>(Bq_t)` | `cb_m_new`(pop) | `cb_m` | advance running max |
| L. recip(l) | helper | `unary` (`Recip`) | eltwise_convenience.hpp:126; eltwise_math.inl:47 | `unary<Recip<>, cb_l, cb_recip_l>(Bq_t)` | `cb_l`(pop) | `cb_recip_l` | after kv loop |
| M. O/l | helper | `mul` | eltwise_convenience.hpp:93 | `mul<cb_o, cb_recip_l, cb_out, BroadcastDim::Col>(Bq_t·d_t)` | `cb_o`(pop),`cb_recip_l` | `cb_out` | final normalize; drains persistent O |
| free Q | raw_api | `cb_pop_front(cb_q, Bq_t·d_t)` | (CB API) | once at unit end | `cb_q` | — | retained Q never popped by matmul |
| reader/writer tile I/O | raw_api | `noc_async_read_tile`/`noc_async_write_tile` + `TensorAccessor` | tensor_accessor.md | per-tile DRAM↔L1 | — | — | standard interleaved access |

### Helpers considered and rejected (raw-API fallbacks)

1. **`compute_kernel_hw_startup` / `mm_block_init` (boot init)** — raw because they *are* the boot primitives the helper library requires the caller to issue exactly once; `matmul_block`'s `InitMode::Short` explicitly does **not** issue `hw_configure`-bearing init (matmul_block_helpers.hpp:138 — "Callers MUST issue mm_block_init() … exactly ONCE at the very top"). No helper covers boot init.
2. **`cb_pop_front(cb_q)` at unit end** — raw because the QKᵀ `matmul_block` uses `WaitAndRetainOnLastBlock` (matmul_block_helpers.hpp:99 InputPolicy doc) so it deliberately never pops `cb_q`; freeing the retained operand at unit end is a one-line CB op with no helper equivalent. Using `WaitAndPopPerKBlock` instead is *provably incorrect* here — it would consume Q after the first KV-chunk, breaking reuse across the kv loop.
3. **reader/writer `noc_async_*_tile`** — raw because data-movement tile transfers are not compute-helper territory; the kernel_lib helpers are all compute-thread (TRISC) primitives.

All other phases use helpers. `streaming_reduce_helpers.hpp::accumulate_reduce` (streaming_reduce_helpers.hpp:79) was considered for the running max/sum but **rejected** (file:line streaming_reduce_helpers.hpp:47–92): it implements `acc ← reduce(block) ⊕ acc` with a single pool op, but Flash Attention's running stats require the *correction* coupling (`l ← exp(m_old−m_new)·l + rowsum`, and `O ← exp(m_old−m_new)·O + PV`) that needs both the pre- and post-update max simultaneously — a recurrence `accumulate_reduce` cannot express (it overwrites the accumulator and exposes no `m_old` for the correction term). The explicit reduce + BinaryMax + exp decomposition is used instead.

## Compute Phases

| # | Operation | Helper? | Input CB (state) | Output CB | CB State After |
|---|-----------|---------|------------------|-----------|----------------|
| 0a | scaler prep ×2 | yes | — | `cb_scaler_max`,`cb_scaler_sum` | scalers resident for whole kernel |
| 0b | init m=−1e30, l=0, O=0 | yes | — | `cb_m`,`cb_l`,`cb_o` | running stats seeded |
| 0c | scale Q | yes | `cb_q_in` | `cb_q` | `cb_q` resident (retained across kv loop); `cb_q_in` freed |
| **kv-loop j = 0 … Nkv−1** | | | | | |
| A | QKᵀ (transpose) | yes | `cb_q`(retained), `cb_k_in` | `cb_scores` | `cb_q` still resident; `cb_k_in` freed |
| B | + mask (custom only) | yes | `cb_scores`,`cb_mask_in` | `cb_scores` | `cb_mask_in` freed |
| C | rowmax | yes | `cb_scores`(no-pop), `cb_scaler_max` | `cb_m_cur` | `cb_scores` still held |
| D | m_new = max(m, m_cur) | yes | `cb_m`(held), `cb_m_cur` | `cb_m_new` | `cb_m` (old) still held; `cb_m_cur` freed |
| E | P = exp(scores − m_new) | yes | `cb_scores`(pop), `cb_m_new`(held) | `cb_p` | `cb_scores` freed; `cb_m_new` held |
| F | rowsum | yes | `cb_p`(no-pop), `cb_scaler_sum` | `cb_l_cur` | `cb_p` still held for PV |
| G | corr = exp(m − m_new) | yes | `cb_m`(pop old), `cb_m_new`(pop) | `cb_corr` | old `cb_m` & `cb_m_new` freed |
| H | l ← corr·l + l_cur | yes | `cb_corr`(held), `cb_l`, `cb_l_cur` | `cb_l` | `cb_l` updated; `cb_l_cur` freed; corr held |
| I | PV matmul | yes | `cb_p`(pop), `cb_v_in` | `cb_pv` | `cb_p`, `cb_v_in` freed |
| J1 | O_resc = corr·O | yes | `cb_o`(pop), `cb_corr`(pop) | `cb_o_resc` | old `cb_o`, `cb_corr` freed |
| J2 | O ← O_resc + PV | yes | `cb_o_resc`, `cb_pv` | `cb_o` | running O updated & re-held; `cb_pv` freed |
| K | m ← m_new | yes | `cb_m_new`(pop) | `cb_m` | running max advanced |
| **end loop** | | | | | |
| L | recip(l) | yes | `cb_l`(pop) | `cb_recip_l` | `cb_l` freed |
| M | O_final = O · (1/l) | yes | `cb_o`(pop), `cb_recip_l` | `cb_out` | persistent `cb_o` drained; output ready |
| Z | free Q | raw | `cb_q`(pop) | — | unit complete, all CBs empty |

For `j == 0` the recurrence is self-correcting: `m = −1e30` ⇒ `corr = exp(−1e30 − m_new) = 0`, so `O = 0·corr + PV = PV` and `l = 0·corr + l_cur = l_cur`. No special-casing required (implementer may instead branch on `j==0` to skip the correction multiplies as a perf option).

## Broadcast Verification

| Phase | Op | CB_A valid region | CB_B valid region | Broadcast Dim |
|-------|-----|-------------------|-------------------|---------------|
| B (mask) | add | `cb_scores` `[Bq_t,Bkv_t]` All | `cb_mask_in` `[Bq_t,Bkv_t]` All | None |
| D | BinaryMax | `cb_m` `[Bq_t,1]` Col0 | `cb_m_cur` `[Bq_t,1]` Col0 | None (both column vectors) |
| E | sub | `cb_scores` `[Bq_t,Bkv_t]` All | `cb_m_new` `[Bq_t,1]` Col0 | **Col** (per-row max bcast across width) |
| G | sub | `cb_m` `[Bq_t,1]` Col0 | `cb_m_new` `[Bq_t,1]` Col0 | None |
| H (mul) | mul | `cb_corr` `[Bq_t,1]` Col0 | `cb_l` `[Bq_t,1]` Col0 | None |
| H (add) | add | `cb_l` `[Bq_t,1]` Col0 | `cb_l_cur` `[Bq_t,1]` Col0 | None |
| J1 | mul | `cb_o` `[Bq_t,d_t]` All | `cb_corr` `[Bq_t,1]` Col0 | **Col** (per-row corr bcast across d_t) |
| J2 | add | `cb_o_resc` `[Bq_t,d_t]` All | `cb_pv` `[Bq_t,d_t]` All | None |
| M | mul | `cb_o` `[Bq_t,d_t]` All | `cb_recip_l` `[Bq_t,1]` Col0 | **Col** (per-row 1/l bcast across d_t) |

`BroadcastDim::Col` is correct for every per-row (REDUCE_ROW → `[Bq_t,1]`) operand broadcast across the width of a `[Bq_t, *]` tile (eltwise_chain.hpp:418–430: REDUCE_ROW result is column-shaped and broadcasts via `Col`).

## Reduce Direction Verification

| Logical reduce | Tile ReduceDim | Output valid region | Downstream BroadcastDim | ReduceInputBlockShape |
|----------------|----------------|---------------------|-------------------------|------------------------|
| row-max over S_kv block | REDUCE_ROW | `[Bq_t,1]` Col0 | Col (phase E) | `of(Bq_t, Bkv_t)` |
| row-sum over S_kv block | REDUCE_ROW | `[Bq_t,1]` Col0 | Col (phase M) | `of(Bq_t, Bkv_t)` |

Softmax reduces along the key (width) axis of the `[Bq_t, Bkv_t]` score block ⇒ `REDUCE_ROW` (collapses columns, yields one value per query row). Both scalers use value `1.0` (MAX and SUM need no averaging) but are prepared with pool-type-aware fills (`prepare_reduce_scaler<…, PoolType::MAX/SUM, ReduceDim::REDUCE_ROW>`) because MAX (Row-0 fill) and SUM-REDUCE_ROW (Col-0 fill) require different tile fill patterns.

## Key Risks and Gotchas

- **Score block sizing is load-bearing.** `cb_scores`, `cb_p` are sized `Bq_t·Bkv_t` — never `Sq_t·Skv_t`. Any code path that grows them to full S×S violates the op's defining constraint.
- **Q must stay resident.** QKᵀ uses `num_k_blocks=1` + `in0_policy=WaitAndRetainOnLastBlock` so the matmul never pops `cb_q`; the kernel pops `cb_q` exactly once at unit end. Using `WaitAndPopPerKBlock` would consume Q after KV-chunk 0 and corrupt every later chunk.
- **`cb_scores` and `cb_p` must survive their no-pop reduce.** rowmax (C) and rowsum (F) use `ReduceInputPolicy::WaitUpfrontNoPop` because the same tiles feed exp (E) and PV (I) respectively. A pop-per-tile policy would drain them before the consumer runs → hang.
- **Both pre- and post-update max needed.** `cb_m` (old) and `cb_m_new` must coexist: `cb_m` is held through phase G (corr) and only popped there; `cb_m_new` is held through phase K. Freeing either early breaks the correction term.
- **Running stats are persistent, single-buffered, in-place.** `cb_m`,`cb_l`,`cb_o` are seeded once (phase 0b) and updated in place each iteration (pop→re-push). They are NOT double-buffered; the recurrence is strictly sequential.
- **Reduce scalers are bf16 and pool-type-aware.** `cb_scaler_max`/`cb_scaler_sum` are bf16, prepared with the `<PoolType, ReduceDim>` overload, value 1.0, and stay resident (reduce waits but does not pop them).
- **Scale folded into Q.** `scale = scale or 1/sqrt(D)` resolved on host, baked as float-bits CT arg, applied via `MulUnary` to Q once per unit — so it is inside the softmax max/exp/sum as required, at minimal cost (Q-block ≪ score blocks).
- **DEST budget.** Output matmul subblocks (`out_subblock_h·out_subblock_w`) must be ≤ 4 under `fp32_dest_acc_en=True`, ≤ 8 otherwise. For large `D` reduce `Bq_t`/`Bkv_t` to keep block CBs within L1.
- **fp32_dest_acc_en honored.** Threaded from `compute_kernel_config` into the `ComputeConfigDescriptor`; never forced to True. fp32+False is an op EXCLUSION (rejected), not silently corrected.

## Refinement contracts (armed by later SUPPORTED additions)

- **`mask_mode=causal`** (adds `causal` to SUPPORTED): derive the triangular −inf bias on-device from `is_causal` (no caller tensor, no materialized full mask). Per (Q-chunk, KV-chunk): skip KV-chunks entirely in the future (no QKᵀ/softmax/PV — the block-skip ≈ halves KV work); pass-through KV-chunks entirely in the past unmasked; apply a per-element triangular −inf only to the diagonal-straddling block (generated into a `cb_scores`-shaped buffer, added before phase C). Add EXCLUSION `{mask_mode: causal, attention_kind: cross}` (causal requires S_q==S_kv) and `ValueError` when `is_causal` + `attn_mask` both set.
- **`kv_heads_mode=gqa/mqa`**: same kernel; reader computes `h_kv = h / group` with `group = H_q/H_kv > 1`. No compute change.
- **`alignment=h/w_non_aligned`**: partial last tiles need reduce partial-scalers (`prepare_partial_reduce_scalers` / `ReducePartialScaler::last_tile_at`) so masked/padding lanes don't pollute row-max / row-sum; head-dim non-alignment pads `D` to a tile.
- **`dtype=float32/bfloat8_b`**: per-CB dtypes follow input; fp32 halves DEST tile budget (subblocks ≤ 4); EXCLUSION `{dtype: float32, fp32_dest_acc_en: False}` arms.

## Structural impossibilities (note for feature_spec.py — pipeline mode)

`feature_spec.py` already declares `INVALID = []` correctly: SDPA is TILE-only (no ROW_MAJOR in TARGET), so the canonical `{bf8b, ROW_MAJOR}` cell is vacuous. No additional structural impossibilities identified.
