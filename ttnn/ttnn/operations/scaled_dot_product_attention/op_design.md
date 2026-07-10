# Operation Design: scaled_dot_product_attention (Flash Attention)

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (matmul → online-softmax → matmul) |
| Goal | Compute `softmax(Q·Kᵀ·scale [+ mask]) · V` using the Flash-Attention algorithm: tile over the sequence dimension and accumulate the weighted output with online softmax (running max + running sum), so the full `S_q × S_kv` score matrix is never materialized (O(S) memory). |
| Math | `O[b,h,:,:] = softmax( (Q[b,h]·K[b,h]ᵀ)·scale + M ) · V[b,h]`, softmax over the `S_kv` (last) axis. `M` = additive mask (custom tensor, on-device causal triangle, or none). |
| Mode | Hybrid (two tiled matmuls bridged by an on-line-softmax recurrence — not a single derivative helper) |
| References | Tri Dao, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"; helper headers cited per phase in **API Mapping**; feature universe: `eval/golden_tests/scaled_dot_product_attention/feature_spec.py`; entry-signature/exports contract: `eval/golden_tests/scaled_dot_product_attention/axes.py:22-32`. |

The load-bearing constraint: the per-`(Q-block, KV-block)` score tile is `B_q × B_kv` and lives only in a circular buffer sized to that block. No CB and no DRAM buffer is ever sized `S_q × S_kv`.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `query` | `ttnn.Tensor` `(B, H_q, S_q, D)` | yes | 4D, TILE, bf16 (Phase 0) | — | tensor |
| `key` | `ttnn.Tensor` `(B, H_kv, S_kv, D)` | yes | 4D, TILE, bf16; `H_q % H_kv == 0` | — | tensor |
| `value` | `ttnn.Tensor` `(B, H_kv, S_kv, D)` | yes | 4D, TILE, bf16; same shape as `key` | — | tensor |
| `attn_mask` | `ttnn.Tensor` or `None` | no | `(B,1,S_q,S_kv)` or `(B,H_q,S_q,S_kv)`, additive, same dtype | `None` | tensor (optional) |
| `is_causal` | `bool` | no | mutually exclusive with `attn_mask` | `False` | RT flag (refinement) |
| `scale` | `float` or `None` | no | any finite; `None` ⇒ `1/√D` | `None` | CT (bit-packed uint32) |
| `compute_kernel_config` | `ttnn.ComputeConfigDescriptor` or `None` | no | resolved through `default_compute_kernel_config()` | `None` | host-only |

Derived (host): `Dt = D/32`, `Sq_t = S_q/32`, `Skv_t = S_kv/32`, `q_chunk_t` / `kv_chunk_t` (compile-time chunk tile counts, see Work Distribution), `scale_bits = struct.unpack("I", struct.pack("f", scale))[0]`.

### Precision axes (`fp32_dest_acc_en` + `dtype`)

- `default_compute_kernel_config()` returns `ttnn.ComputeConfigDescriptor(math_fidelity=HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)` — a fresh descriptor per call, exported from the op package (imported by `axes.py:25`). Resolve `None` through it; never inline the default elsewhere.
- `validate()` reads `fp32_dest_acc_en` off the resolved config and **rejects `float32 + fp32_dest_acc_en=False`** via `EXCLUSIONS` (maxed input + non-maxed accumulation is lossy). `bfloat16`/`bfloat8_b` accept both flag values. The op must honor the caller's flag (read it, reject the illegal pair) — never silently force `True`.
- `math_fidelity` / `math_approx_mode` are not axes; accept any value.
- Both matmuls run with K-accumulation across `Dt`/`kv_chunk_t` tiles in DEST, so bf16 requires `fp32_dest_acc_en=True` (Phase 0 corner) for correctness — matmul header precision note, `matmul_block_helpers.hpp:253-259`.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | Q `(B,H_q,S_q,D)`, K/V `(B,H_kv,S_kv,D)`; optional mask `(B,1 or H_q,S_q,S_kv)` |
| Dtype | bf16 (Phase 0); fp32 / bf8_b refinements |
| Layout | TILE only (no ROW_MAJOR in TARGET) |
| Memory | DRAM or L1, interleaved |

### Output

| Property | Value |
|----------|-------|
| Shape | `(B, H_q, S_q, D)` (same as Q) |
| Dtype | same as Q (bf16 Phase 0) |
| Layout | TILE |
| Memory | DRAM interleaved (or caller `memory_config`) |

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one **Q-block** = `(b, h_q, q_chunk)`; inner loop streams every KV-block of that `(b, h_q)`. |
| Grid | compute grid up to device 2D grid (e.g. `8×8`); `num_q_blocks = B · H_q · ceil(Sq_t / q_chunk_t)`. |
| Per-core work | `ttnn.split_work_to_cores(num_q_blocks, grid)` → contiguous run of Q-blocks per core. Each Q-block: load Q chunk once, loop `num_kv_blocks = ceil(Skv_t / kv_chunk_t)` KV-blocks, run the online-softmax recurrence, normalize, write the output chunk. |
| Chunk sizing | `q_chunk_t = min(Sq_t, 4)`, `kv_chunk_t = min(Skv_t, 4)` (≈128 rows/keys, both compile-time). `Dt` (head_dim) is **never** chunked — the full head dim is the matmul contraction for QᵀK and the output width for PV. |
| Remainder | Q/KV sequence not a multiple of the chunk: the **last** chunk carries the residual tile count (`Sq_t % q_chunk_t`, `Skv_t % kv_chunk_t`); block shapes are recomputed per chunk (Phase 0 is tile-aligned so every chunk is a whole number of tiles). Per-`(b,h_q)` block count divides cleanly; residual Q-blocks across cores are handled by `split_work_to_cores`. |
| GQA/MQA head map | reader maps `h_kv = h_q / (H_q / H_kv)`; K/V pages addressed with `h_kv`. Pure reader-side addressing — compute is head-agnostic. `H_q == H_kv` ⇒ mha, `H_kv == 1` ⇒ mqa, else gqa. |
| Cross-attention | `S_q != S_kv` only changes the two loop bounds; no other change. |

Inter-Tensix communication: **none**. Each Q-block is fully independent (its own running `m/l/O`); no multicast/semaphores. Every core reads Q/K/V/mask from DRAM via `TensorAccessor` and writes its output chunk back to DRAM.

## Circular Buffers

Page size for every tile CB = `tile_size(dtype)` (bf16 ≈ 2048 B). "full block" num_pages means the CB holds the whole named block (sequential compute helpers own all TRISCs — see `ttnn-cb-memory-fundamentals.md`). Persistent CBs live across the entire KV loop of a Q-block.

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_q` | 0 | tile | `q_chunk_t·Dt` | bf16 | reader | compute (QKᵀ in0) | held for whole KV loop (pre-scaled in place, reused every KV-block) |
| `cb_k` | 1 | tile | `2·kv_chunk_t·Dt` | bf16 | reader | compute (QKᵀ in1) | per KV-block, double-buffered |
| `cb_v` | 2 | tile | `2·kv_chunk_t·Dt` | bf16 | reader | compute (PV in1) | per KV-block, double-buffered |
| `cb_mask` | 3 | tile | `2·q_chunk_t·kv_chunk_t` | bf16 | reader (custom mask only) | compute (score add) | per KV-block; absent when `mask_mode=none` |
| `cb_scaler_max` | 8 | tile | 1 | bf16 | reader | compute (MAX reduce) | whole kernel |
| `cb_scaler_sum` | 9 | tile | 1 | bf16 | reader | compute (SUM reduce) | whole kernel |
| `cb_l_new` | 10 | tile | `q_chunk_t` | fp32/bf16 | compute | compute | per KV-block temp |
| `cb_pv` | 11 | tile | `q_chunk_t·Dt` | fp32/bf16 | compute (PV matmul) | compute (O update) | per KV-block temp |
| `cb_o_run` | 12 | tile | `q_chunk_t·Dt` | fp32/bf16 | compute | compute / writer path | **persistent** running output |
| `cb_o_new` | 13 | tile | `q_chunk_t·Dt` | fp32/bf16 | compute | compute (commit) | per KV-block temp |
| `cb_l_inv` | 14 | tile | `q_chunk_t` | fp32/bf16 | compute (Recip) | compute (final mul) | per Q-block (final) |
| `cb_out` | 16 | tile | `2·q_chunk_t·Dt` | bf16 | compute (normalize) | writer | per Q-block output |
| `cb_scores` | 24 | tile | `q_chunk_t·kv_chunk_t` | fp32/bf16 | compute (QKᵀ matmul) | compute (mask add, MAX reduce, P chain) | per KV-block; full block (reduce needs it resident) |
| `cb_probs` | 25 | tile | `q_chunk_t·kv_chunk_t` | fp32/bf16 | compute (P chain) | compute (SUM reduce, PV matmul) | per KV-block; full block (reused by reduce + matmul) |
| `cb_m_cur` | 26 | tile | `q_chunk_t` | fp32/bf16 | compute (MAX reduce) | compute | per KV-block block-max |
| `cb_m_run` | 27 | tile | `q_chunk_t` | fp32/bf16 | compute | compute | **persistent** running max |
| `cb_m_new` | 28 | tile | `q_chunk_t` | fp32/bf16 | compute | compute (P chain, correction, commit) | per KV-block temp |
| `cb_corr` | 29 | tile | `q_chunk_t` | fp32/bf16 | compute (exp) | compute (l/O rescale) | per KV-block temp |
| `cb_l_cur` | 30 | tile | `q_chunk_t` | fp32/bf16 | compute (SUM reduce) | compute (l update) | per KV-block block-sum |
| `cb_l_run` | 31 | tile | `q_chunk_t` | fp32/bf16 | compute | compute / final | **persistent** running sum |

Intermediate/temp CBs that feed the next compute helper are sized full-block; the running `m/l/O` CBs are held across the KV loop. `cb_k`/`cb_v`/`cb_mask`/`cb_out` are reader↔compute or compute↔writer streaming buffers and are double-buffered.

Matmul `interm_buf`: both matmuls run with `num_k_blocks == 1` (single K-block spanning the whole contraction), so there is **no spill** — pass the matmul's own `out_buf` as the interm placeholder (`matmul_block_helpers.hpp:246-247, 303-309`). No dedicated interm CB.

## API Mapping

Every mechanism has a verified file:line reference. All helpers own their own CB reserve/push/wait/pop unless noted. Caller issues `compute_kernel_hw_startup(...)` once at boot and `mm_block_init()` once at boot (matmul requirement, `matmul_block_helpers.hpp:100-104`).

| Phase | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Requirements |
|-------|------|----------|-----------|------------------------|----------|-----------|--------------|
| Pre-scale Q | helper | `transform_in_place` | `eltwise_convenience.hpp:232` | `<cb_q>(EltwiseShape::tiles(q_chunk_t·Dt), MulUnary<>{scale_bits})` (`MulUnary` `eltwise_scalar.hpp:32`) | `cb_q` | `cb_q` (in place) | scale once per Q-block; owns wait/pop/reserve/push |
| QKᵀ | helper | `matmul_block` | `matmul_block_helpers.hpp:353` | `<transpose=true, packer_l1_acc=false, LastBlockTarget::Out, in0_policy=WaitAndRetainOnLastBlock>`; `MatmulBlockShape::of(q_chunk_t/sb_h, kv_chunk_t/sb_w, sb_h, sb_w, Dt, 1)` (`:157`) | `cb_q` (in0, retained), `cb_k` (in1) | `cb_scores` (out=interm placeholder) | K contraction = `Dt` in one K-block; transpose flips K tiles so out = `Q·Kᵀ` = `[q_chunk_t × kv_chunk_t]`. `in0` retained so Q survives all KV-blocks (`InputPolicy` `:77`) |
| Mask add (custom) | helper | `add` | `eltwise_convenience.hpp:56` | `<cb_scores, cb_mask, cb_scores, BroadcastDim::None>(EltwiseShape::grid(q_chunk_t, kv_chunk_t))` | `cb_scores`, `cb_mask` | `cb_scores` (in place) | additive mask, per-element; in-place via pop-before-reserve. Compile-time elided when `mask_mode=none` |
| Block row-max | helper | `reduce` | `reduce_helpers_compute.hpp:421` | `<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_scores, cb_scaler_max, cb_m_cur, ReduceInputPolicy::WaitUpfrontNoPop>(ReduceInputBlockShape::of(q_chunk_t, kv_chunk_t))` | `cb_scores`, `cb_scaler_max` | `cb_m_cur` | reduces W (keys) → `q_chunk_t` col-vectors; `WaitUpfrontNoPop` keeps `cb_scores` for the P chain (`:107`, example `:372-378`) |
| Running-max merge (j>0) | helper | `binary_sfpu` | `eltwise_convenience.hpp:158` | `<BinaryMax<>, cb_m_run, cb_m_cur, cb_m_new>(q_chunk_t)` (`BinaryMax` `eltwise_binary_sfpu.hpp:72`) | `cb_m_run`, `cb_m_cur` | `cb_m_new` | elementwise max of two col-vectors |
| P = exp(S − m) | helper | `eltwise_chain` | `eltwise_chain.hpp:576` | `BinaryFpu<cb_scores, cb_m_new, BinaryFpuOp::Sub, BroadcastDim::Col, …>` (`:500`) → `Exp<>{}` (`eltwise_math.hpp:21`) → `PackTile<cb_probs, …>` (`:535`); shape `grid(q_chunk_t, kv_chunk_t)` | `cb_scores`, `cb_m_new` (`cb_m_cur` on j==0) | `cb_probs` | fused sub-broadcast + exp in one dst window — the softmax `(x−max)→exp` pattern (`eltwise_chain.hpp:49-53` / BroadcastDim note `:420-424`). Consumes/pops `cb_scores` |
| Block row-sum | helper | `reduce` | `reduce_helpers_compute.hpp:421` | `<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_probs, cb_scaler_sum, cb_l_cur, ReduceInputPolicy::WaitUpfrontNoPop>(ReduceInputBlockShape::of(q_chunk_t, kv_chunk_t))` | `cb_probs`, `cb_scaler_sum` | `cb_l_cur` | `WaitUpfrontNoPop` keeps `cb_probs` for PV matmul |
| Correction = exp(m_run − m_new) (j>0) | helper | `eltwise_chain` | `eltwise_chain.hpp:576` | `BinaryFpu<cb_m_run, cb_m_new, BinaryFpuOp::Sub, BroadcastDim::None>` → `Exp<>{}` → `PackTile<cb_corr>`; shape `tiles(q_chunk_t)` | `cb_m_run`, `cb_m_new` | `cb_corr` | col-vector fused sub+exp; computed **before** `cb_m_run` is overwritten |
| l update = corr·l_run + l_cur (j>0) | helper | `eltwise_chain` | `eltwise_chain.hpp:576` | `BinaryFpu<cb_corr, cb_l_run, BinaryFpuOp::Mul>` → `DestReuseBinary<cb_l_cur, BinaryFpuOp::Add, DestReuseType::DEST_TO_SRCA>` (`:515`) → `PackTile<cb_l_new>`; `tiles(q_chunk_t)` | `cb_corr`, `cb_l_run`, `cb_l_cur` | `cb_l_new` | fused multiply-accumulate on col-vectors |
| PV matmul | helper | `matmul_block` | `matmul_block_helpers.hpp:353` | `<transpose=false, LastBlockTarget::Out>`; `MatmulBlockShape::of(q_chunk_t/sb_h, Dt/sb_w, sb_h, sb_w, kv_chunk_t, 1)` | `cb_probs` (in0), `cb_v` (in1) | `cb_pv` (out=interm placeholder) | K contraction = `kv_chunk_t`; out = `[q_chunk_t × Dt]`. Consumes/pops `cb_probs`, `cb_v` |
| O update = corr·O_run + PV (j>0) | helper | `eltwise_chain` | `eltwise_chain.hpp:576` | `BinaryFpu<cb_o_run, cb_corr, BinaryFpuOp::Mul, BroadcastDim::Col, …>` → `DestReuseBinary<cb_pv, BinaryFpuOp::Add, DEST_TO_SRCA>` → `PackTile<cb_o_new>`; `grid(q_chunk_t, Dt)` | `cb_o_run`, `cb_corr`, `cb_pv` | `cb_o_new` | corr broadcast across D columns, then add PV |
| Commit running state | helper | `copy` | `eltwise_convenience.hpp:180` | `copy<cb_m_new, cb_m_run>(q_chunk_t)`, `copy<cb_l_new, cb_l_run>(q_chunk_t)`, `copy<cb_o_new, cb_o_run>(q_chunk_t·Dt)` | new CBs | run CBs | end-of-KV-block ping-pong commit. On j==0 the block results (`cb_m_cur`/`cb_l_cur`/`cb_pv`) are copied straight into the run CBs (no correction) |
| Final 1/l | helper | `unary` | `eltwise_convenience.hpp:136` | `<Recip<>, cb_l_run, cb_l_inv>(q_chunk_t)` (`Recip` `eltwise_math.hpp:33`) | `cb_l_run` | `cb_l_inv` | reciprocal of running sum |
| Final normalize | helper | `mul` | `eltwise_convenience.hpp:94` | `<cb_o_run, cb_l_inv, cb_out, BroadcastDim::Col>(grid(q_chunk_t, Dt))` | `cb_o_run`, `cb_l_inv` | `cb_out` | `O / l` broadcast across D → output chunk |
| Reduce scalers | helper | `calculate_and_prepare_reduce_scaler` | `reduce_helpers_dataflow.hpp:84` | `<cb_scaler_max, PoolType::MAX, ReduceDim::REDUCE_ROW>()` and `<cb_scaler_sum, PoolType::SUM, ReduceDim::REDUCE_ROW>()` | — | `cb_scaler_max`, `cb_scaler_sum` | pool-type-aware overload (fill layout differs per PoolType); reader runs each once at start (`:26-38`) |
| Q/K/V/mask read, O write | raw_api | `TensorAccessor` | `tech_reports/tensor_accessor/tensor_accessor.md` | `TensorAccessorArgs<...>` per tensor; `noc_async_read` / `noc_async_write` | DRAM → CBs / `cb_out` → DRAM | see below | see "Helpers considered and rejected" |

### Helpers considered and rejected (raw-API reader/writer)

- **Phase: DRAM ↔ L1 dataflow for Q/K/V/mask/output.** This is pure address generation and NoC transfer, not a compute phase — no `kernel_lib` compute helper covers it. The compute helpers (`matmul_block`, `reduce`, `eltwise_chain`) all consume/produce CBs that are already resident in L1; none read DRAM. `TensorAccessor` is the sanctioned dataflow mechanism. `mcast_pipe` (`mcast_pipe.hpp`) was considered for K/V distribution but **rejected**: each Q-block streams its own K/V from DRAM independently with no inter-core sharing in this design (no rectangle of receiver cores consuming one broadcast), so `SenderPipe`/`ReceiverPipe` would add a semaphore handshake with no reuse benefit. Per-core interleaved DRAM reads are correct and simpler.

No compute phase uses a raw API — every compute phase maps to a helper above.

## Compute Phases

Per Q-block, after `compute_kernel_hw_startup` + `mm_block_init` (boot, once):

| # | Operation | Helper? | Input CB (tiles, state) | Output CB (tiles) | CB State After |
|---|-----------|---------|-------------------------|-------------------|----------------|
| 0 | Pre-scale Q | `transform_in_place` | `cb_q` (`q_chunk_t·Dt`, fronted) | `cb_q` | `cb_q` holds scaled Q, held for whole KV loop |
| — | **for each KV-block j:** | | | | |
| 1 | QKᵀ | `matmul_block<transpose=true>` | `cb_q` (retained), `cb_k` (`kv_chunk_t·Dt`) | `cb_scores` (`q_chunk_t·kv_chunk_t`) | `cb_q` retained; `cb_k` popped |
| 2 | Mask add (custom only) | `add` in place | `cb_scores`, `cb_mask` | `cb_scores` | `cb_mask` popped |
| 3 | Block row-max | `reduce<MAX,ROW,WaitUpfrontNoPop>` | `cb_scores` (kept), `cb_scaler_max` | `cb_m_cur` (`q_chunk_t`) | `cb_scores` still resident |
| 4 | m_new (j>0) / passthrough (j==0) | `binary_sfpu<BinaryMax>` (j>0) | `cb_m_run`, `cb_m_cur` | `cb_m_new` | on j==0 use `cb_m_cur` directly as m_new |
| 5 | P = exp(S−m_new) | `eltwise_chain` (Sub·Col→Exp→Pack) | `cb_scores`, `cb_m_new` | `cb_probs` (`q_chunk_t·kv_chunk_t`) | `cb_scores` popped |
| 6 | Block row-sum | `reduce<SUM,ROW,WaitUpfrontNoPop>` | `cb_probs` (kept), `cb_scaler_sum` | `cb_l_cur` (`q_chunk_t`) | `cb_probs` still resident |
| 7 | Correction (j>0) | `eltwise_chain` (Sub→Exp→Pack) | `cb_m_run`, `cb_m_new` | `cb_corr` | old `cb_m_run` consumed before commit |
| 8 | l update (j>0) | `eltwise_chain` (Mul→Add-reuse→Pack) | `cb_corr`, `cb_l_run`, `cb_l_cur` | `cb_l_new` | — |
| 9 | PV | `matmul_block<transpose=false>` | `cb_probs`, `cb_v` (`kv_chunk_t·Dt`) | `cb_pv` (`q_chunk_t·Dt`) | `cb_probs`, `cb_v` popped |
| 10 | O update (j>0) | `eltwise_chain` (Mul·Col→Add-reuse→Pack) | `cb_o_run`, `cb_corr`, `cb_pv` | `cb_o_new` | — |
| 11 | Commit | `copy` ×3 | `cb_m_new`/`cb_l_new`/`cb_o_new` (j>0) or `cb_m_cur`/`cb_l_cur`/`cb_pv` (j==0) | `cb_m_run`/`cb_l_run`/`cb_o_run` | running state updated; per-block temps freed |
| — | **end KV loop** | | | | `cb_o_run`, `cb_l_run` hold final unnormalized O and l |
| 12 | 1/l | `unary<Recip>` | `cb_l_run` | `cb_l_inv` | — |
| 13 | Normalize | `mul<…,Col>` | `cb_o_run`, `cb_l_inv` | `cb_out` (`q_chunk_t·Dt`) | `cb_out` ready for writer |

**j==0 handling (numerically exact, no `-inf` sentinel):** the first KV-block skips phases 4/7/8/10 and instead sets `m_run←m_cur`, `l_run←l_cur`, `o_run←PV` directly (phase 11 branch). This avoids `exp(-∞)` / `0·∞` entirely, keeping the recurrence exact.

## Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast Dim |
|-------|-----|-------------------|-------------------|---------------|
| Mask add | `add` | `cb_scores` `[q_chunk_t,kv_chunk_t]` = All | `cb_mask` `[q_chunk_t,kv_chunk_t]` = All | None |
| m_new merge | `binary_sfpu` max | `cb_m_run` Col0 | `cb_m_cur` Col0 | None (both col-vectors, same shape) |
| P = exp(S−m) | `eltwise_chain` Sub | `cb_scores` = All | `cb_m_new` = Col0 (REDUCE_ROW result) | **Col** (broadcast col-vector across keys) |
| Correction | `eltwise_chain` Sub | `cb_m_run` Col0 | `cb_m_new` Col0 | None |
| l update | Mul + Add | `cb_corr`/`cb_l_run`/`cb_l_cur` all Col0 | — | None |
| O update | Mul | `cb_o_run` = All `[q_chunk_t,Dt]` | `cb_corr` = Col0 | **Col** (correction across D) |
| Normalize | `mul` | `cb_o_run` = All | `cb_l_inv` = Col0 | **Col** (1/l across D) |

Rule check: every REDUCE_ROW result (`cb_m_cur`, `cb_m_new`, `cb_l_run`, `cb_l_inv`) is column-shaped (Col0) and is broadcast back with `BroadcastDim::Col` — matches the eltwise_chain BroadcastDim convention (`eltwise_chain.hpp:420-424`).

## Reduce Direction Verification

| Logical Dim | Tile ReduceDim | Output Valid Region | BroadcastDim | ReduceInputBlockShape | EltwiseShape |
|-------------|----------------|---------------------|--------------|-----------------------|--------------|
| softmax over keys (`S_kv`, last axis) | `REDUCE_ROW` (reduces W) | Col0, `q_chunk_t` tiles | `Col` | `of(q_chunk_t, kv_chunk_t)` | `grid(q_chunk_t, kv_chunk_t)` |

Softmax is always over the last axis (`S_kv`); the score block is laid out `[q_chunk_t (rows) × kv_chunk_t (cols)]`, so reducing W (`REDUCE_ROW`) is the per-query reduction. Single fixed direction — no multi-dim reduce.

## Validation Contract (op file)

`INPUT_TAGGERS` (op-local, single source of truth; imported by `axes.py` and `feature_spec.py`):
- `tag_alignment(inputs, axes)` on Q's `(S_q, D)` = `inputs[0][-2:]`: `"tile_aligned"` (both %32==0), `"w_non_aligned"` (D%32≠0), `"h_non_aligned"` (D aligned, S_q%32≠0).
- `tag_attention_kind(inputs, axes)`: `"self"` if `inputs[0][-2] == inputs[1][-2]` else `"cross"`.
- `tag_kv_heads(inputs, axes)`: `H_q = inputs[0][1]`, `H_kv = inputs[1][1]` → `"mha"` (equal), `"mqa"` (`H_kv==1`), else `"gqa"`.

Derived-in-`validate()` (from kwargs, NOT taggers): `mask_mode ∈ {none, custom, causal}` from `is_causal`/`attn_mask` (raise `ValueError` if both set — mutually exclusive); `scale_mode ∈ {auto, explicit}` from `scale is None`; `fp32_dest_acc_en` from the resolved compute config.

Phase 0 `SUPPORTED`: `dtype=[bfloat16]`, `fp32_dest_acc_en=[True, False]`, `layout=[TILE]`, `alignment=[tile_aligned]`, `attention_kind=[self, cross]`, `kv_heads_mode=[mha, gqa, mqa]`, `mask_mode=[none, custom]`, `scale_mode=[auto, explicit]`.

Phase 0 `EXCLUSIONS`: `{dtype: float32, fp32_dest_acc_en: False}` (arms with the dtype refinement; kept declared so it holds when fp32 lands). Raises `ExcludedCell`.

Tensor-shape contract checks (raise `ValueError`/`RuntimeError`, independent of axes): rank≠4; `Q.D != K.D`; `K.S != V.S` or `K.H != V.H`; `Q.B != K.B`; `H_q % H_kv != 0`; mask shape not `(B, 1|H_q, S_q, S_kv)`.

`validate()` is the first line of the public entry point. Refusals use `UnsupportedAxisValue` / `ExcludedCell` from `ttnn.operations._op_contract`.

## Refinements & Rules (arm when SUPPORTED grows toward TARGET)

- **`mask_mode=causal`** (native `is_causal`, no mask tensor): derive the triangular mask **on-device**, block-by-block, never materialize the full `S_q×S_kv` mask. Three regions per `(Q-block, KV-block)`: past blocks unmasked; **future blocks are whole-tile −∞ and SHOULD be skipped** (don't run QKᵀ/softmax/PV — the ≈½ KV-work causal win); the diagonal-straddling block gets a per-element triangular −∞ generated on-device (SFPU fill / iota compare). Applied additively onto scores before the row-max (same additive primitive as custom mask). When causal is added to SUPPORTED, also declare `EXCLUSION {mask_mode: causal, attention_kind: cross}` (causal requires `S_q == S_kv`) and raise `ValueError` when `is_causal=True` and `attn_mask is not None`.
- **`dtype` refinements** (`float32`, `bfloat8_b`): expose `compute_kernel_config`; keep the `{float32, fp32_dest_acc_en=False}` EXCLUSION; fp32 requires HiFi4+fp32_dest_acc; bf16 K>1 requires HiFi2+fp32_dest_acc (`matmul_block_helpers.hpp:253-259`). CBs already fp32-capable (temps declared fp32-friendly).
- **`alignment` refinements** (`w_non_aligned`, `h_non_aligned`): partial last tile along S/D; use the partial-scaler pair for reductions (`ReducePartialScaler::last_tile_at(1)` + `calculate_and_prepare_partial_reduce_scalers`, `reduce_helpers_dataflow.hpp:149`) and mask padded score columns to −∞ before softmax.

**Structural impossibilities:** reviewed `feature_spec.py` — `INVALID=[]` is correct (TILE-only universe ⇒ the bf8_b+ROW_MAJOR canonical rule is vacuous). No additional INVALID candidates.

## Key Risks and Gotchas

- **Flash-Attention invariant:** `cb_scores`/`cb_probs` are sized `q_chunk_t·kv_chunk_t` (per-block), never `Sq_t·Skv_t`. No CB or DRAM buffer is sequence-squared. This is the op's reason to exist — do not "simplify" by building the whole score matrix.
- **Persistent running state:** `cb_m_run`, `cb_l_run`, `cb_o_run` must survive the entire KV loop of a Q-block; they are committed (phase 11) each block and read next block. Do not double-buffer/pop them inside the loop.
- **`cb_q` retention:** QKᵀ uses `in0_policy=WaitAndRetainOnLastBlock` so Q is not popped and is reused for every KV-block; pop `cb_q` only at Q-block end.
- **Reduce keeps its input:** both reductions use `WaitUpfrontNoPop` so the score/prob block stays resident for the following P chain / PV matmul; the consuming helper does the pop.
- **Scaler CBs are pool-type-aware bf16:** two separate scalers (`MAX` and `SUM`, both `REDUCE_ROW`) prepared with `calculate_and_prepare_reduce_scaler<cb, PoolType, ReduceDim>` — the MAX and SUM fill patterns differ; do not share one CB or use the legacy `prepare_reduce_scaler<cb>` overload.
- **j==0 special-case (not `-inf`):** first KV-block copies block results into the running state; this keeps the recurrence numerically exact and avoids `exp(-∞)`/`0·∞`. A fully-masked row cannot occur in Phase 0 (custom masks are finite, none = no mask).
- **DEST budget:** with `fp32_dest_acc_en=True`, matmul output subblock ≤ 4 tiles (`DEST_AUTO_LIMIT` halved). Pick `out_subblock_h·out_subblock_w ≤ 4` for both matmuls; eltwise chains process one/`block_size` tiles per dst window and stay within budget.
- **Single K-block matmuls:** `num_k_blocks==1` means the whole contraction (`Dt` for QKᵀ, `kv_chunk_t` for PV) accumulates in DEST in one block — requires `fp32_dest_acc_en=True` for bf16 correctness, and lets `interm_buf = out_buf` (no spill CB).
- **GQA/MQA is reader-only:** compute never sees head broadcasting; the reader maps `h_q → h_kv`. Getting the K/V page base wrong silently reads the wrong head.
- **Scale is applied once to Q** (pre-scale, phase 0), not to the `S_q×S_kv` scores — cheaper (`q_chunk_t·Dt` vs `q_chunk_t·kv_chunk_t` per block) and equivalent.
- **CB sync:** every producer push count equals the consumer wait count — matmul/reduce/eltwise helpers each own their CB ops; do not wrap them with extra `cb_*` calls.
