# Operation Design: scaled_dot_product_attention (Flash Attention)

## Blocking Model (decided FIRST — everything downstream references this)

Flash Attention computes `output = softmax(Q·Kᵀ·scale + mask)·V` **without ever
materializing the `S_q × S_kv` score matrix**. It is parameterized as a blocking
scheme: tile Q along `S_q` into blocks, stream all `(K,V)` blocks along `S_kv`
once per Q block, and fold each score block into per-row running statistics
(`m` = running max, `l` = running sum, `O` = running weighted output) via the
online-softmax recurrence. The score block lives in a CB sized `B_q × B_kv`
tiles — never the whole matrix. This is the load-bearing constraint of the op.

### Axis table

| Axis | Character | Block-factor knob | Phase-1 value | Core-assignment | Later unlock |
|------|-----------|-------------------|---------------|-----------------|--------------|
| **Batch** `B` | **independent** — each batch's attention is isolated | 1 batch per work unit (not sub-chunked) | 1 | folded into the work-unit id and spread across the grid | knob-turn |
| **Query heads** `H` | **independent** for MHA — each head isolated; the KV head is **reuse-shared** across the Q heads that map to it under GQA/MQA | 1 head per work unit | 1 | folded into the work-unit id and spread across the grid | knob-turn (MHA); reuse **scheme-change** (GQA/MQA KV mcast) |
| **Query seq** `S_q` | **independent** — online softmax is per-Q-row; each Q block's output rows are computed in isolation | `Sq_chunk_t` (Q-block height in tiles) | `min(Sq_t, 4)`, L1-capped (coarsest that fits) | `n_q_chunks` per head spread across the grid; each core loops its assigned q-chunks | knob-turn |
| **KV seq** `S_kv` | **dependent** — the softmax normalization and `O` accumulation are a reduction spanning all of `S_kv` (a result spans the axis) | `Skv_chunk_t` (KV-block height in tiles) | `min(Skv_t, 4)`, L1-capped | single-core per work unit — the online accumulate is a cheap **sequential** combine within the core | **scheme-change** (cross-core flash-decode combine of partial `m,l,O`) |
| **Head dim** `D` | contraction of `Q·Kᵀ` (a matmul-internal reduction) and free dim of `P·V` output — not a cross-core split candidate | matmul `num_k_blocks` / `in0_block_k` | `num_k_blocks=1` (whole `D` in one K-block) | not core-split; handled by matmul K-blocking | knob-turn (raise `num_k_blocks` under L1 pressure) |

`Sq_t = ceil(S_q/32)`, `Skv_t = ceil(S_kv/32)`, `Dt = ceil(D/32)` — always `ceil`, per-head (alignment-aware even though phase-1 only exercises tile-aligned shapes).

### Buffer-depth knobs (per streaming CB)

| CB | Depth knob | Phase-1 value | Rationale |
|----|-----------|---------------|-----------|
| `cb_k_in`, `cb_v_in`, `cb_mask_in` | `KV_DEPTH` | 2 (double-buffer) | overlap NoC read of KV chunk *j+1* with compute of chunk *j* (FlashAttention paper §3.2.2; catalog `double_buffer` — depth = 2×block, sweet spot 4–8 tiles in flight) |
| `cb_out` | `OUT_DEPTH` | 2 | overlap writer drain with compute of next q-chunk |
| `cb_q_in`, `cb_q_scaled`, `cb_out_accum`, `cb_row_max`, `cb_row_sum`, `cb_corr` | resident (depth 1) | 1 | reused/accumulated across the whole KV loop of one q-chunk |
| `cb_scores`, `cb_exp`, `cb_pv` | full-block (depth 1) | 1 | sequential handoff between compute helpers (both own all 3 TRISCs — cannot pipeline) |

### Single source of truth for the block factors

`Sq_chunk_t`, `Skv_chunk_t`, `KV_DEPTH` are computed **once** on the host by a
`_fit_l1(D, dtype, fp32_dest_acc_en, grid)` routine and threaded as compile-time
args. Every CB page count and every kernel loop bound is **derived from these
three values** — no block dimension is restated as a second literal, so a later
perf/OOM refinement turns the knob in one place. `_fit_l1` starts from the coarse
targets `Sq_chunk_t=Skv_chunk_t=4` (128-row chunks — the amortizing granularity,
not the 1-tile minimum), `KV_DEPTH=2`, and shrinks in order
`Skv_chunk_t → KV_DEPTH(2→1) → Sq_chunk_t` until the working-set bytes fit the L1
budget (≈1.4 MB). Large `D` (e.g. `D=1024 → Dt=32`) is exactly the case that
shrinks the KV chunk; small `D` keeps the coarse defaults.

### Bandwidth ranking of candidate splits (qualitative, no ns)

- **Split `B · H · q-chunks` (independent) — CHOSEN primary split.** No cross-core
  combine at all. Each core reads its Q chunk once and streams its `(K,V)` for one
  `(batch,head)`; those reads are inherent to the algorithm. Fewest bytes moved
  for **prefill** (large `S_q`), which is the primary use case (long context S=2k–8k).
- **Split `S_kv` across cores (dependent) — lamp.** Each core reduces a slice of
  KV and produces partial `(m,l,O)`; a cross-core combine merges them (Flash-Decode /
  logical width-shard). Adds partial-`O`+stats NoC traffic. This is the available
  parallelism **only when `B·H·n_q_chunks` under-fills the grid** (e.g. `B=H=1`,
  small `S_q`, huge `S_kv` cross-attention). Reachable whether or not the caller
  pre-shards.
- **Reuse-shared KV under GQA/MQA — lamp.** The Q heads mapping to one KV head
  re-read the same `K,V` from DRAM per head. Multicasting `K,V` once to the group
  of cores that own those Q heads eliminates the re-reads (paper §5 future-work).
  Correctness does not need it (per-head DRAM re-read is correct); this is bandwidth only.

### Lamps (scheme-changes phase-1 leaves room for)

1. **Causal masking (`mask_mode=causal`).** On-device triangular mask; **block-skip**
   whole future KV chunks (≈½ the KV work) and apply a per-element diagonal mask only
   on the straddling block. Reachable because phase-1's KV loop already iterates
   per-chunk and the additive-mask primitive is already wired (the `custom` path);
   causal adds a skip predicate + on-device mask generation. Arms EXCLUSION
   `{mask_mode:causal, attention_kind:cross}` (causal requires `S_q==S_kv`) and the
   `is_causal ∧ attn_mask` ValueError.
2. **Flash-Decode (`S_kv` cross-core combine).** Split the dependent KV axis across
   cores when the independent axes under-fill the grid. Reachable because the
   recurrence already maintains partial `(m,l,O)` — the cross-core combine is the
   *same* rescale math applied across cores via a NoC reduction
   (`mcast_pipe.hpp` / semaphores). Surfaces as TARGET `memory_layout ∈
   {WIDTH_SHARDED, BLOCK_SHARDED}` → **scheme-change**.
3. **GQA/MQA KV multicast.** Swap per-head DRAM re-read for an mcast receive of the
   shared KV head. Reachable because the KV read is already a distinct dataflow phase.
4. **Resident (non-streaming) fast path.** When the whole `S_kv` provably fits L1
   (`Skv_t·Dt·bytes ≤ budget`), a dual-path dispatched on that fits-in-L1 predicate
   loads all K/V once and skips the chunk loop (streaming path is the fallback).

### memory_layout (TARGET axis) → scheme mapping

| memory_layout | Meaning | Cost class |
|---------------|---------|------------|
| `INTERLEAVED` | phase-1 baseline; block geometry read from DRAM | phase-1 |
| `HEIGHT_SHARDED` | logical `S_q`/head shard **pinned** by caller, data pre-placed in L1; work stays per-core | **knob-turn** (CB/placement change; honor shard as the per-core block — no sub-chunk unless it exceeds L1) |
| `WIDTH_SHARDED`, `BLOCK_SHARDED` | logical `S_kv`/dependent split; partials cross cores | **scheme-change** (Lamp 2 — flash-decode combine) |

---

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (matmul → softmax → matmul, single op, no DRAM round-trip of intermediates) |
| Goal | Exact scaled-dot-product attention via the FlashAttention-2 algorithm: tiled, online-softmax, O(S) memory. The `S_q×S_kv` scores are never materialized whole. |
| Math | `S = (Q·scale)·Kᵀ` ; `S += mask` (custom/causal) ; `P = softmax(S, dim=-1)` ; `O = P·V` — computed incrementally over KV blocks with a running `(m,l,O)` recurrence. |
| Mode | Hybrid (custom fused kernel) |
| References | `tech_reports/FlashAttention/FlashAttention.md` (parallelization + online softmax + causal load-balancing); Tri Dao "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"; `.claude/references/ttnn-cb-memory-fundamentals.md`; catalog `ttnn/ttnn/operations/examples/master.md` entries `double_buffer`, `compute_block_size`, `matmul_output_subblock`, `reduce_block`, `row_reduce_accumulate`, `compute_fusion`, `reader_placement`, `eltwise_l1_vs_dest_accumulate`; `eval/op_template.py`; registry ops rms_norm/softmax. |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `query` | `ttnn.Tensor` `(B,H,S_q,D)` | yes | 4D, TILE, bf16 (phase-0) | — | tensor |
| `key` | `ttnn.Tensor` `(B,H_kv,S_kv,D)` | yes | 4D, TILE, bf16; `H_kv∈{H, 1, divisor of H}` | — | tensor |
| `value` | `ttnn.Tensor` `(B,H_kv,S_kv,D)` | yes | same shape as `key` | — | tensor |
| `attn_mask` | `ttnn.Tensor` `(B,1,S_q,S_kv)` or `(B,H,S_q,S_kv)` | no (kw-only) | additive, bf16, TILE; mutually exclusive with `is_causal` | `None` | tensor (optional) |
| `is_causal` | `bool` (kw-only) | no | `True`/`False`; mutually exclusive with `attn_mask` | `False` | → `mask_mode` (CT flag) |
| `scale` | `float` (kw-only) | no | any finite; `None` → `1/sqrt(D)` | `None` | RT arg (packed as u32) |
| `compute_kernel_config` | `ttnn.ComputeConfigDescriptor` (kw-only) | no | resolves via `default_compute_kernel_config()` | `None` | drives `math_fidelity`, `fp32_dest_acc_en` |
| `memory_config` | `ttnn.MemoryConfig` (kw-only) | no | DRAM/L1 interleaved (phase-0) | DRAM interleaved | output placement |

`default_compute_kernel_config()` — the single source of truth for the maxed-out
corner: `math_fidelity=HiFi4`, `fp32_dest_acc_en=True`, `math_approx_mode=False`
(mirrors rms_norm). `None` resolves through it in **both** `validate()` and the
entry point; the golden axis-tagger reads the same factory.

### Support axes and `validate()` contract

`validate(query, key, value, attn_mask, is_causal, scale, compute_kernel_config)`
is the **first line** of the entry point. It (1) raises `ValueError`/`RuntimeError`
for structural violations, then (2) builds the axes dict exactly as the golden
harness does and gates on `SUPPORTED` then `EXCLUSIONS` (raising the typed
`UnsupportedAxisValue` / `ExcludedCell` from `ttnn.operations._op_contract`).

Structural checks (raise before axis gating):
- rank ≠ 4 for any of Q/K/V → `ValueError`.
- `Q.D != K.D` (head_dim mismatch) → `ValueError`.
- `K.shape != V.shape` (K/V seq or head mismatch) → `ValueError`.
- `Q.B != K.B` (batch mismatch) → `ValueError`.
- `H_q % H_kv != 0` (illegal GQA/MQA ratio) → `ValueError`.
- `attn_mask` present with incompatible dims (`mask.S_q != Q.S_q`, `mask.S_kv != K.S_kv`, mask head dim ∉ {1, H}) → `ValueError`.
- `is_causal=True` **and** `attn_mask is not None` → `ValueError` (Torch's mutual-exclusion rule).

Axes (names match `feature_spec.py` TARGET and the op's INPUT_TAGGERS):

| Axis | Source | TARGET values |
|------|--------|---------------|
| `dtype` | `query.dtype` | `[float32, bfloat16, bfloat8_b]` |
| `fp32_dest_acc_en` | `bool(cfg.fp32_dest_acc_en)` | `[True, False]` |
| `layout` | `query.layout` | `[TILE_LAYOUT]` |
| `alignment` | `tag_alignment` on Q's `(S_q, D)`: `tile_aligned` (both %32==0) / `w_non_aligned` (D%32≠0) / `h_non_aligned` (D aligned, S_q not) | `[tile_aligned, w_non_aligned, h_non_aligned]` |
| `attention_kind` | `tag_attention_kind`: `self` if `Q.S_q==K.S_kv` else `cross` | `[self, cross]` |
| `kv_heads_mode` | `tag_kv_heads`: `mha` if `H_q==H_kv`; `mqa` if `H_kv==1`; else `gqa` | `[mha, gqa, mqa]` |
| `mask_mode` | kwargs: `causal` if `is_causal`; `custom` if `attn_mask is not None`; else `none` | `[none, custom, causal]` |
| `scale_mode` | kwargs: `auto` if `scale is None` else `explicit` | `[auto, explicit]` |

Phase-0 SUPPORTED (implementer's claim; verifier files each `TARGET − SUPPORTED`
value as a refinement): `dtype=[bfloat16]`, `fp32_dest_acc_en=[True]`,
`layout=[TILE]`, `alignment=[tile_aligned]`, `attention_kind=[self, cross]`,
`kv_heads_mode=[mha, gqa, mqa]` (correctness is a trivial KV-head index mapping —
see Work Distribution; the mcast optimization is Lamp 3), `mask_mode=[none, custom]`,
`scale_mode=[auto, explicit]`. Phase-0 EXCLUSIONS: none.

Refinements arm these EXCLUSIONS (do **not** declare them phase-0): with the
`dtype`/precision refinement, `{dtype: float32, fp32_dest_acc_en: False}`
(maxed input + non-maxed accumulation — lossy, refused, honoring the caller's
flag rather than silently forcing `True`); with the causal refinement,
`{mask_mode: causal, attention_kind: cross}`.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | Q `(B,H,S_q,D)`; K,V `(B,H_kv,S_kv,D)`; optional mask `(B,{1,H},S_q,S_kv)` |
| Dtype | bfloat16 (phase-0); float32/bfloat8_b are refinements |
| Layout | TILE |
| Memory | DRAM or L1, interleaved (phase-0) |

### Output

| Property | Value |
|----------|-------|
| Shape | `(B,H,S_q,D)` — same as Q |
| Dtype | bfloat16 (phase-0) |
| Layout | TILE |
| Memory | DRAM interleaved (or `memory_config`) |

## Dataflow Strategy

Data path per work unit `(b, h, q_chunk)` on one core:

```
DRAM Q[b,h, qc-block, :]  ──reader──► cb_q_in ──compute(prescale)──► cb_q_scaled (resident)
DRAM K[b,kv_head, :, :]   ──reader──► cb_k_in  (streamed, KV_DEPTH=2)
DRAM V[b,kv_head, :, :]   ──reader──► cb_v_in  (streamed, KV_DEPTH=2)
DRAM mask[b,{0,h}, qc, :] ──reader──► cb_mask_in (custom only, streamed)
                          reader fills cb_scaler(1.0), cb_scale(scale) once
compute: for kv_chunk j in 0..n_kv_chunks-1:
    cb_q_scaled·cb_k_inⱼ ─matmul(Kᵀ)─► cb_scores
    (custom) cb_scores += cb_mask_inⱼ
    online-softmax recurrence updates cb_row_max, cb_row_sum, cb_out_accum
    cb_expⱼ·cb_v_inⱼ ─matmul─► cb_pv ; cb_out_accum = α·cb_out_accum + cb_pv
compute: cb_out_accum · (1/cb_row_sum) ─► cb_out
cb_out ──writer──► DRAM output[b,h, qc-block, :]
```

Between RISCs: reader (NCRISC) → compute (3 TRISCs) via input CBs; compute →
writer (BRISC) via `cb_out`. All intermediates stay in L1 CBs — the scores
matrix is never assembled beyond one `Sq_chunk_t × Skv_chunk_t` block.

**Unlocked-scheme contract (Lamp 2, flash-decode — described though phase-1 does
not use it):** when `S_kv` is split across a group of cores, each core computes
partial `(mᵏ, lᵏ, Oᵏ)` over its KV slice, then a designated reducer core (or a
ring) combines them: `m = max_k mᵏ`; `l = Σ_k exp(mᵏ−m)·lᵏ`;
`O = Σ_k exp(mᵏ−m)·Oᵏ`, normalized by `l`. Partials are shipped via
`mcast_pipe.hpp` (`SenderPipe::send` / `ReceiverPipe::receive`) with a semaphore
handshake; ordering is fixed by the sender rotation the host emits
(`host/mcast_host.hpp`). This is the same rescale math the intra-core recurrence
already runs, lifted across cores.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one `(batch b, query-head h, q-chunk qc)` — an independent block of `Sq_chunk_t` Q tile-rows |
| Grid | `device.compute_with_storage_grid_size()` — a **runtime parameter**, so the core count stays tunable |
| Per-core work | `split_work_to_cores(grid, total_work, row_wise=True)` → two core groups with `per_core_1`/`per_core_2` work units; `row_wise=True` spreads the core line across the DRAM-facing axis (catalog `reader_placement`: ~2.2–2.8× over a column line) |
| Remainder | handled by the two-group split; the last q-chunk of a head may be a partial block when `Sq_t % Sq_chunk_t ≠ 0` — kernel processes only the valid tile-rows |

Formulae (all `ceil`, per-head — alignment-ready):
```
Sq_t = ceil(S_q/32);  Skv_t = ceil(S_kv/32);  Dt = ceil(D/32)
n_q_chunks  = ceil(Sq_t  / Sq_chunk_t)         # q-blocks per head
n_kv_chunks = ceil(Skv_t / Skv_chunk_t)        # KV loop length
total_work  = B * H * n_q_chunks
# decode a work-unit index w:
b  = w // (H * n_q_chunks)
r  = w %  (H * n_q_chunks)
h  = r  // n_q_chunks
qc = r  %  n_q_chunks
kv_head = h // (H // H_kv)                       # MHA→h ; GQA→group ; MQA→0
```
The `kv_head` division is the whole of GQA/MQA correctness: the reader points its
K/V TensorAccessor at `kv_head` instead of `h`. No new loop, no new algorithm.

**Regime selection (>1 compute regime → pinned + regime-pinned tests):** the only
phase-1 regime axis is `has_mask`, a compile-time flag:
`has_mask = (mask_mode == "custom")` i.e. `attn_mask is not None`. `has_mask=false`
skips the mask-add and the `cb_mask_in` reads; `has_mask=true` streams and adds the
mask block before the row-max. Predicate is stated here; the acceptance test pins
**both** `none` and `custom`. (The `causal` regime — block-skip + on-device mask —
arms with the causal refinement.)

## Circular Buffers

Page size = one 32×32 tile of the stated format. `bf16 ≈ 2048 B`, `fp32 ≈ 4096 B`.
All `Num Pages` are functions of the block/buffer knobs only — **no CB grows with a
whole-op dimension** (`S_q`, `S_kv` never appear).

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_q_in` | 0 | tile | `Sq_chunk_t · Dt` | bf16 | reader | compute | resident per q-chunk |
| `cb_k_in` | 1 | tile | `Skv_chunk_t · Dt · KV_DEPTH` | bf16 | reader | compute | streamed |
| `cb_v_in` | 2 | tile | `Skv_chunk_t · Dt · KV_DEPTH` | bf16 | reader | compute | streamed |
| `cb_mask_in` | 3 | tile | `Sq_chunk_t · Skv_chunk_t · KV_DEPTH` | bf16 | reader | compute | streamed (custom only) |
| `cb_scaler` | 4 | tile | `1` | bf16 | reader | compute | whole op (1.0 for MAX & SUM) |
| `cb_scale` | 5 | tile | `1` | bf16 | reader | compute | whole op (resolved attention scale) |
| `cb_q_scaled` | 24 | tile | `Sq_chunk_t · Dt` | bf16 | compute | compute | resident per q-chunk |
| `cb_scores` | 25 | tile | `Sq_chunk_t · Skv_chunk_t` | bf16 | compute | compute | per KV chunk |
| `cb_exp` | 26 | tile | `Sq_chunk_t · Skv_chunk_t` | bf16 | compute | compute | per KV chunk |
| `cb_row_max` | 27 | tile | `Sq_chunk_t` | fp32 | compute | compute | resident per q-chunk (running `m`) |
| `cb_row_sum` | 28 | tile | `Sq_chunk_t` | fp32 | compute | compute | resident per q-chunk (running `l`) |
| `cb_pv` | 29 | tile | `Sq_chunk_t · Dt` | fp32 | compute | compute | per KV chunk (PV partial) |
| `cb_out_accum` | 30 | tile | `Sq_chunk_t · Dt` | fp32 | compute | compute | resident per q-chunk (running `O`) |
| `cb_corr` | 31 | tile | `Sq_chunk_t` | fp32 | compute | compute | per KV chunk (chunk-max / `α` scratch) |
| `cb_out` | 16 | tile | `Sq_chunk_t · Dt · OUT_DEPTH` | bf16 | compute | writer | streamed to DRAM |

**Ownership:** every CB has exactly one producer *kernel* and one consumer *kernel*.
`cb_q_scaled/scores/exp/row_max/row_sum/pv/out_accum/corr` are compute-internal
(compute produces, compute consumes — normal intermediates, no second kernel touches
them). `cb_scores` is scaled/masked in place by compute only; no dataflow kernel
reads it. Scale is applied via a **separate `cb_q_scaled`** (not an in-place write
of `cb_q_in`) precisely so `cb_q_in` keeps its single producer (reader).

**Sizing rationale:** streaming KV CBs carry `KV_DEPTH` blocks (double-buffer);
sequential-helper intermediates (`cb_scores`,`cb_exp`,`cb_pv`) hold a full block
because consecutive helpers each own all 3 TRISCs and cannot pipeline; running
statistics/accumulators are single resident blocks; `cb_scaler`/`cb_scale` are one
tile. `cb_row_max`/`cb_row_sum`/`cb_corr` are `Sq_chunk_t` tiles because a
`REDUCE_ROW` result is column-shaped (`rows × 1`).

## API Mapping

Every mechanism has a verified file:line. Helpers under
`ttnn/cpp/ttnn/kernel_lib/`. Every compute phase uses a helper; no raw-API compute
fallback is needed.

| Phase | Type | Function | File:Line | Template Params / Args (block/chunk knobs **bold**) | Input CB | Output CB | Requirements |
|-------|------|----------|-----------|------------------------------------------------------|----------|-----------|--------------|
| boot | helper | `compute_kernel_hw_startup` + `mm_block_init` | `matmul_block_helpers.hpp:100-104,320-322` | — | — | — | once at top of compute `kernel_main` before any helper |
| pre-scale Q | helper | `mul` (scalar bcast) | `eltwise_convenience.hpp:81-98` | `Bcast=BroadcastDim::Scalar`; **`EltwiseShape::tiles(Sq_chunk_t·Dt)`** | `cb_q_in`, `cb_scale` | `cb_q_scaled` | once per q-chunk; folds `scale` so mask adds on scaled scores uniformly |
| QKᵀ | helper | `matmul_block` (transpose) | `matmul_block_helpers.hpp:334-366` | `transpose=true`, `last_block_target=Out`; **`MatmulBlockShape::of(in0_sub=Sq_chunk_t, in1_sub=Skv_chunk_t, out_sub_h, out_sub_w, in0_block_k=Dt, num_k_blocks=1)`** | `cb_q_scaled`, `cb_k_in` | `cb_scores` | `num_k_blocks=1` ⇒ no spill (pass `cb_scores` as interm placeholder, `:304-309`); out subblock ≤ DEST (4 fp32 tiles) |
| mask add | helper | `add` | `eltwise_convenience.hpp:43-60` | `Bcast=None`; **`EltwiseShape::tiles(Sq_chunk_t·Skv_chunk_t)`** | `cb_scores`, `cb_mask_in` | `cb_scores` (in place) | custom mode only; applied **before** row-max |
| chunk row-max | helper | `reduce<MAX, REDUCE_ROW>` | `reduce_helpers_compute.hpp:471-487` | `PoolType::MAX`, `ReduceDim::REDUCE_ROW`; ids `cb_scores/cb_scaler/cb_corr`; **`ReduceInputBlockShape::row(Skv_chunk_t, Sq_chunk_t)`** | `cb_scores`, `cb_scaler` | `cb_corr` | scaler=1.0; running max carried via `BinaryMax` (Accumulate CB path static-asserts on MAX+REDUCE_ROW, `:255-259`) |
| running max + `α` | helper | `binary_sfpu<BinaryMax>` + `unary<Exp>` | `eltwise_convenience.hpp:146-165,126-139`; `eltwise_math.hpp:21-22` | **`EltwiseShape::tiles(Sq_chunk_t)`** | `cb_corr`,`cb_row_max` | `cb_row_max`,`cb_corr(α)` | `m_new=max(m_old,chunk_max)`; `α=exp(m_old−m_new)`; keep `m_old` until `α` computed |
| `P=exp(S−m)` | helper | `sub`(bcast col) + `unary<Exp>` (fused chain) | `eltwise_chain.hpp:710-711`; `eltwise_convenience.hpp:62-79,126-139`; `Exp` `eltwise_math.hpp:21-22` | `Bcast=BroadcastDim::Col`; **`EltwiseShape::grid(Sq_chunk_t, Skv_chunk_t)`** | `cb_scores`,`cb_row_max` | `cb_exp` | `REDUCE_ROW` result is column-shaped ⇒ `Col` bcast (`eltwise_chain.hpp:522-534`) |
| chunk row-sum | helper | `reduce<SUM, REDUCE_ROW>` | `reduce_helpers_compute.hpp:471-487` | `PoolType::SUM`, `ReduceDim::REDUCE_ROW`; **`ReduceInputBlockShape::row(Skv_chunk_t, Sq_chunk_t)`** | `cb_exp`, `cb_scaler` | scratch → `cb_row_sum` | `l_new = α·l_old + rowsum(P)` (rescale then add) |
| rescale `O` | helper | `mul`(bcast col) | `eltwise_convenience.hpp:81-98` | `Bcast=BroadcastDim::Col`; **`EltwiseShape::tiles(Sq_chunk_t·Dt)`** | `cb_out_accum`,`cb_corr(α)` | `cb_out_accum` | before adding PV partial |
| PV | helper | `matmul_block` | `matmul_block_helpers.hpp:334-366` | `transpose=false`; **`MatmulBlockShape::of(in0_sub=Sq_chunk_t, in1_sub=Dt, out_sub_h, out_sub_w, in0_block_k=Skv_chunk_t, num_k_blocks=1)`** | `cb_exp`, `cb_v_in` | `cb_pv` | contraction over `Skv_chunk_t`; `num_k_blocks=1` ⇒ no spill |
| accumulate `O` | helper | `add` | `eltwise_convenience.hpp:43-60` | **`EltwiseShape::tiles(Sq_chunk_t·Dt)`** | `cb_out_accum`,`cb_pv` | `cb_out_accum` | `O += PV` (fp32 accumulator) |
| normalize | helper | `unary<Recip>` + `mul`(bcast col) | `eltwise_math.hpp:33-34`; `eltwise_convenience.hpp:81-98` | `Bcast=BroadcastDim::Col`; **`EltwiseShape::tiles(Sq_chunk_t·Dt)`** | `cb_row_sum`,`cb_out_accum` | `cb_out` | after all KV chunks: `O = O·(1/l)`, packed to bf16 |
| scaler fill | helper | `calculate_and_prepare_reduce_scaler` (pool-type-aware) | `reduce_helpers_dataflow.hpp:97-99` | `<cb_scaler, PoolType::MAX, ReduceDim::REDUCE_ROW>` and `<…, PoolType::SUM, …>` | — | `cb_scaler` | reader-side; 1.0 for MAX/SUM (one 1.0 tile serves both) |

Reader/writer use `TensorAccessor` (raw dataflow API — no helper covers DRAM
addressing) per `tech_reports/tensor_accessor/tensor_accessor.md` and the
CB-fundamentals TensorAccessor pattern. `mcast_pipe.hpp` is **not** used phase-1
(reserved for Lamps 2 & 3).

**Helpers considered and rejected:** none — every compute phase maps to a helper
above. Softmax/flash-attention has no single turnkey helper, so it is assembled
from `matmul_block` + `reduce` + `eltwise_*` as shown (confirmed by the reduce-helper
doc's own softmax worked example, `reduce_helpers_compute.hpp:434-457`).

## Compute Phases

Sequential execution for one q-chunk. `m` initialized to `−∞`, `l=0`, `O=0`.

| # | Operation | Helper? | Input CB (tiles, state) | Output CB (tiles) | CB State After |
|---|-----------|---------|-------------------------|-------------------|----------------|
| 0 | boot init | yes | — | — | engine ready |
| 1 | pre-scale Q | `mul` | `cb_q_in` (`Sq_chunk_t·Dt`, filled), `cb_scale` (1) | `cb_q_scaled` (`Sq_chunk_t·Dt`) | `cb_q_scaled` resident; `cb_q_in` popped |
| 2 | QKᵀ (per KV chunk *j*) | `matmul_block` | `cb_q_scaled` (reused, retained), `cb_k_in` (`Skv_chunk_t·Dt`) | `cb_scores` (`Sq_chunk_t·Skv_chunk_t`) | `cb_k_inⱼ` popped |
| 3 | mask add (custom) | `add` | `cb_scores`, `cb_mask_in` (`Sq_chunk_t·Skv_chunk_t`) | `cb_scores` | `cb_mask_inⱼ` popped |
| 4 | chunk row-max | `reduce<MAX,ROW>` | `cb_scores`, `cb_scaler` | `cb_corr` (`Sq_chunk_t`) | scores retained |
| 5 | update `m`, form `α` | `binary_sfpu<BinaryMax>`, `unary<Exp>` | `cb_corr`, `cb_row_max` (old `m`) | `cb_row_max` (new `m`), `cb_corr` (`α`) | `m` updated |
| 6 | `P=exp(S−m)` | `sub`+`unary<Exp>` chain | `cb_scores`, `cb_row_max` | `cb_exp` (`Sq_chunk_t·Skv_chunk_t`) | `cb_scores` popped |
| 7 | chunk row-sum + `l` update | `reduce<SUM,ROW>`, `mul`+`add` | `cb_exp`, `cb_scaler`, `cb_row_sum`, `cb_corr(α)` | `cb_row_sum` (new `l`) | `l` updated |
| 8 | rescale `O` by `α` | `mul` (bcast col) | `cb_out_accum`, `cb_corr(α)` | `cb_out_accum` | `O` rescaled |
| 9 | PV | `matmul_block` | `cb_exp`, `cb_v_in` (`Skv_chunk_t·Dt`) | `cb_pv` (`Sq_chunk_t·Dt`) | `cb_exp`,`cb_v_inⱼ` popped |
| 10 | accumulate `O` | `add` | `cb_out_accum`, `cb_pv` | `cb_out_accum` | end of KV chunk *j*; loop to 2 |
| 11 | normalize (after loop) | `unary<Recip>`, `mul` (bcast col) | `cb_row_sum`, `cb_out_accum` | `cb_out` (`Sq_chunk_t·Dt`) | `cb_out` pushed to writer; accumulators reset for next q-chunk |

## Broadcast Verification

| Phase | Op | CB_A valid region | CB_B valid region | Broadcast Dim |
|-------|-----|-------------------|-------------------|---------------|
| pre-scale Q (1) | `mul` | `cb_q_in` `[Sq_chunk_t·Dt]` All | `cb_scale` scalar (Row0/Col0 single tile) | Scalar |
| mask add (3) | `add` | `cb_scores` `[Sq_chunk_t, Skv_chunk_t]` All | `cb_mask_in` `[Sq_chunk_t, Skv_chunk_t]` All | None |
| `P=exp(S−m)` (6) | `sub` | `cb_scores` `[Sq_chunk_t, Skv_chunk_t]` All | `cb_row_max` `[Sq_chunk_t]` Col0 (REDUCE_ROW result) | Col |
| rescale `O` (8) | `mul` | `cb_out_accum` `[Sq_chunk_t, Dt]` All | `cb_corr(α)` `[Sq_chunk_t]` Col0 | Col |
| accumulate `O` (10) | `add` | `cb_out_accum` `[Sq_chunk_t, Dt]` All | `cb_pv` `[Sq_chunk_t, Dt]` All | None |
| normalize (11) | `mul` | `cb_out_accum` `[Sq_chunk_t, Dt]` All | `cb_row_sum⁻¹` `[Sq_chunk_t]` Col0 | Col |

## Key Risks and Gotchas

- **The O(S) constraint is load-bearing.** `cb_scores`/`cb_exp` are sized
  `Sq_chunk_t · Skv_chunk_t` (one block) and **must not** grow with `S_q` or
  `S_kv`. Any CB whose size references a whole-sequence dimension is a bug.
- **fp32 accumulators are mandatory for numerical exactness.** `cb_row_max`,
  `cb_row_sum`, `cb_out_accum`, and the `α` correction run in fp32 (phase-0
  `fp32_dest_acc_en=True`). Catalog `row_reduce_accumulate`: bf16 *accumulation*
  error grows with reduction width (13.3 ULP @ W=32) while fp32-DEST stays at
  0.24 ULP — a bf16 accumulator degrades with sequence length and breaks the
  "exact softmax" guarantee. Honor the caller's `fp32_dest_acc_en`; never force it.
- **Running max, not the Accumulate CB path.** `MAX + REDUCE_ROW` via the reduce
  `Accumulate` CB static-asserts (`reduce_helpers_compute.hpp:255-259`). The
  running max must be carried explicitly with `BinaryMax` against the per-chunk
  `reduce<MAX,ROW>` result.
- **Correction ordering.** `α = exp(m_old − m_new)` needs `m_old` **before**
  overwriting the running max; rescale `O` and `l` by `α` **before** folding in
  this chunk's PV / row-sum. First chunk: `m_old=−∞ ⇒ α=0`, which correctly zeroes
  the empty accumulators.
- **Mask add before the max.** Both custom (and later causal) masks are additive
  and applied to the scaled scores **before** the row-max, so masked positions
  fall out of max, exp, and sum together. Never build the full `S_q×S_kv` mask.
- **Scale via pre-scaled Q, one path.** Pre-scaling Q (not folding scale into exp)
  keeps mask handling identical across `none`/`custom`/`causal`; `cb_q_scaled` is a
  distinct CB so `cb_q_in`'s single-producer invariant holds.
- **Reduce scaler format.** `cb_scaler` is bf16 and filled by the pool-type-aware
  `calculate_and_prepare_reduce_scaler<…, PoolType, ReduceDim>` (1.0 for both MAX
  and SUM) — never the legacy `prepare_reduce_scaler<cb>`.
- **DEST budget.** Output subblocks of both matmuls ≤ 4 fp32 tiles (fp32 DEST
  halves the 8-tile bf16 budget); the subblock decomposition is left to the matmul
  tuner — the design exposes only the **block size** and the fp32-DEST flag.
- **Q reuse across the KV loop.** `cb_q_scaled` is retained (not popped) across all
  KV chunks and popped once at end of the q-chunk; the QKᵀ matmul must use an in0
  policy that does not pop it per K-block.
- **Persistent-across-phases data.** `cb_q_scaled`, `cb_row_max`, `cb_row_sum`,
  `cb_out_accum` persist across the entire KV loop of a q-chunk; only reset when the
  next q-chunk begins.

## Structural impossibilities (candidate INVALID — for `/golden-tests`, not this op file)

`feature_spec.py` already declares `INVALID = []` and SDPA is TILE-only, so the
canonical `bf8b + ROW_MAJOR` rule is vacuous. No additional structural
impossibility is proposed. (The `is_causal ∧ attn_mask` combination is a runtime
contract error raised by `validate()`, and `causal + cross` is an EXCLUSION — both
are op-side gates, not test-harness INVALID.)
