# `moe_compute` Pipeline Integration: dispatch → compute → combine → reduce-scatter

> How `ttnn.experimental.moe_compute` is meant to be used in conjunction with the other
> MoE ops, in both **decode** and **prefill** scenarios. Companion to
> [`moe_compute_vs_moe_gpt.md`](./moe_compute_vs_moe_gpt.md).
>
> Sources: `models/common/modules/moe/tt_moe_decode.py`,
> `models/demos/deepseek_v3/tt/moe_optimized.py`, the `all_to_all_dispatch_metadata` /
> `selective_reduce_combine` / `deepseek_moe_fast_reduce_nc_fused` ops, and the
> `deepseek_prefill` op set.

## 1. Summary

`moe_compute` is **only the expert-FFN stage** of a 4-stage, token-routed MoE pipeline. It
never runs alone — it is always bracketed by an all-to-all **dispatch** op upstream and a
**score-weighted combine + reduce-scatter** downstream. Two facts are easy to get wrong and
worth stating up front:

- The production decode/prefill modules call `moe_compute` in **Full mode** (they pass
  `cluster_axis` + `optional_output_tensor` and unpack the 6-tensor return), **not**
  `compute_only`.
- `moe_compute`'s Full-mode combine (`selective_reduce_combine`) performs the **cross-device
  gather/reduce** of each token's *K* expert outputs into a `[K, T, H]` stack. It does
  **not** apply routing scores and does **not** sum over experts — those happen in a
  separate **downstream** op. So `combine_output[k]` is "token *t*'s unweighted output from
  its *k*-th selected expert."

## 2. The canonical op DAG (one MoE layer)

```
                      router / gate  (MoEGate.forward | deepseek_prefill.moe_grouped_topk)
                            │  topk_indices [B,1,1,K], topk_scores [B,1,1,K]
                            ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ 1. ttnn.experimental.all_to_all_dispatch_metadata            │
   │    in : x [D,T,H], topk_indices, topk_scores, expert_mapping │
   │    out: sparse_buffer  [D,T,H]   (tokens routed to this dev) │
   │         expert_indices [D,T,Keff] (all-gathered, L1 drain)   │
   │         expert_scores  [D,T,Keff]                            │
   └─────────────────────────────────────────────────────────────┘
                            │  (3 dispatch outputs + expert_mapping + packed W0/W1 + W2)
                            ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ 2. ttnn.experimental.moe_compute   (Full mode)               │
   │    expert FFN:  act(x·W0)·(x·W1) · W2  per (token, expert)   │
   │    out[3] tilize L1 (scratch, usually deallocated)           │
   │    out[4] matmul output (per-expert; the final out in        │
   │           compute_only mode)                                 │
   │    out[5] combine output [Keff, T, H] ← cross-device gather  │
   │           of each token's K expert outputs (NO scores yet)   │
   └─────────────────────────────────────────────────────────────┘
                            │  combine_output [K, T, H]
                            ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ 3. score-weighted combine over the K experts                 │
   │    (this is where routing scores are applied + summed)       │
   │    pre: deepseek_moe_post_combine_tilize / tilize_with_val_  │
   │         padding  (row-major [K,T,H] → tiled, tile-aligned)   │
   │    • common module:                                          │
   │        deepseek_moe_fast_reduce_nc_fused(                    │
   │            tilized_out, indices, mapping, scores,            │
   │            shared_expert_scale)  → per-token [1,T,H]         │
   │    • deepseek_v3 optimized:                                  │
   │        ttnn.mul(combine_out, scores) then                   │
   │        ttnn.sum / deepseek_moe_fast_reduce_nc over K        │
   └─────────────────────────────────────────────────────────────┘
                            │  per-token output [1,T,H]
                            ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ 4. reduce-scatter over the *replicated* (TP) axis = 1-axis   │
   │    deepseek_moe_reduce_scatter (ring, 8-way) | ttnn.reduce_  │
   │    scatter (generic)  → per-device chunk [1,T,H/num_replic]  │
   └─────────────────────────────────────────────────────────────┘
```

## 3. Role of each surrounding op

| Op | Role relative to `moe_compute` |
|---|---|
| **router / `moe_gate`** (`MoEGate.forward`, or `deepseek_prefill.moe_grouped_topk`) | Produces per-token top-k expert **indices** and **scores**. Runs before dispatch. |
| **`all_to_all_dispatch_metadata`** | The *upstream* op. One op that **both** (a) cross-device routes each token's activations into a per-device **sparse buffer** of the experts living locally, and (b) all-gathers + L1-shards the **indices/scores** to the drain-tilizer core. Its 3 outputs are exactly `moe_compute`'s first 3 inputs. Shared experts are added here as extra local-broadcast slots. |
| **`moe_compute`** | The *compute* stage. Runs the gate/up/down expert matmuls + activation on the sparse buffer. In **Full mode** it also runs the fused `selective_reduce_combine` to gather each token's K expert outputs back across the cluster into a `[K, T, H]` stack — but it does **not** apply scores or sum over experts. |
| **`deepseek_moe_post_combine_tilize`** / `tilize_with_val_padding` | Layout adapter: row-major `[K,T,H]` combine output → tiled, tile-aligned, ready for the reduce op. |
| **`deepseek_moe_fast_reduce_nc_fused`** (common module) **or** `mul`+`sum`/`deepseek_moe_fast_reduce_nc` (deepseek opt) | The *score-weighted expert combine*. Multiplies each routed expert's output by its per-token score and each shared expert by a fixed `shared_expert_scale`, then sums over the K dimension → one vector per token. **This is where the MoE math actually finishes.** |
| **`deepseek_moe_reduce_scatter`** / `ttnn.reduce_scatter` | The *downstream* CCL op. Reduces + scatters the per-token output across the replicated/tensor-parallel axis so each device ends up with `H/num_replicated` of every token. |

### Where the routing scores live (important)

- `moe_compute`'s combine = **cross-device gather/reduce only** (no scores). The
  `selective_reduce_combine` kernels contain no score/scale multiply.
- The **score-weighting + sum-over-K** is always a separate downstream op. So
  `combine_output[k]` is token *t*'s unweighted output from its *k*-th selected expert.

### Weight & mapping setup (host-side, once)

`moe_compute`'s two weight inputs are pre-packed offline with
`ttnn.experimental.prepare_w0_w1_tensor_for_moe_compute` /
`prepare_w2_tensor_for_moe_compute` (+ `_with_bias` variants), then quantized to
`bfloat4_b`. Shared experts are spliced in with `add_shared_expert_weights`.
`expert_mapping` is a replicated `[devices, experts]` tensor shared by **both** the dispatch
and the downstream reduce so they agree on expert→device placement.

## 4. Decode scenario

- One forward step, **`seq_len = 1`**; the per-device token count `batch_per_device` is
  small (≈16–32). The dispatch sparse buffer and all of `moe_compute`'s L1 working set fit,
  so the layer runs as **a single pass** through the DAG above.
- Cross-device sync uses single-use `dispatch_global_semaphore` /
  `combine_global_semaphore` (no double-buffering needed — combine syncs after fully reading
  the dispatch output; dispatch syncs at end of pipeline).
- The final reduce-scatter has a fast fused path (`deepseek_moe_reduce_scatter`) when on a
  ring fabric with an 8-wide replicated axis and 32-token tiles; otherwise the generic
  `ttnn.reduce_scatter`.

Reference: `models/common/modules/moe/tt_moe_decode.py` (`forward()`),
`models/demos/deepseek_v3/tt/moe_optimized.py` (`_fwd_decode_moe` →
`_forward_moe_optimized_ring_impl`).

## 5. Prefill scenario

There are **two distinct strategies**; which one a model uses is a design choice.

### (A) Same `moe_compute` DAG, batch-chunked

`models/demos/deepseek_v3/tt/moe_optimized.py::_fwd_prefill_moe`:

- Identical pipeline to decode, but the token batch is **chunked** (`moe_chunk_size`) and run
  iteratively, because *"`moe_compute` requires just about all of L1"* and prefill has many
  tokens — a full batch's dispatch buffer would overflow L1. Each chunk is padded, run,
  sliced back to its real size, and the results are concatenated.
- Memory configs flip to **DRAM** (decode uses sharded L1); indices/scores are staged in
  DRAM between chunks.
- **seq↔batch interchange**: the A2A dispatch/combine require the leading data-parallel dim
  to equal `num_dispatch_devices`, so for `bs=1` long-sequence prefill the sequence tokens
  are reshaped to occupy the "batch/token" axis (`seq_len=1` is forced;
  see `moe_optimized.py:367`).

### (B) A dedicated prefill MoE op set that does *not* use `moe_compute`

`models/demos/deepseek_v3_d_p/tt/moe/` + the `deepseek_prefill` ops:

- Pipeline: `moe_grouped_topk` (grouped routing) → `masked_bincount` / `offset_cumsum`
  (routing setup) → `deepseek_prefill.dispatch` → `routed_expert_ffn` *(or the
  Blackhole-fused `unified_routed_expert_moe`)* → `shared_expert` →
  `deepseek_prefill.combine` → sum-experts + reduce-scatter.
- Uses **dense, capacity-based per-expert matmuls** (gather tokens per expert by offset →
  dense MLP → scatter back) rather than the sparse ring-matmul of `moe_compute`. More
  natural for the large-token, compute-bound prefill regime and more mesh-flexible (1D/2D,
  padding-aware, shared experts in parallel).
- `models/demos/deepseek_v3_b1/` similarly uses its own fused ops / micro-ops, not
  `moe_compute`.

## 6. Decode vs prefill, in one line

`moe_compute` is the **token-routed, sparse-ring** expert engine. It is the default for
**decode** (few tokens, latency/memory-bound — sparse routing wins), and it *can* serve
**prefill** when batch-chunked, but heavier prefill workloads often use the **dense
grouped-matmul `deepseek_prefill` path** instead. In all cases `moe_compute` is sandwiched
between an A2A **dispatch** upstream and a **score-weighted combine + reduce-scatter**
downstream — it computes the experts, but the routing-score weighting and the across-expert
/ across-device reductions are owned by its neighbours.

## 7. Two concrete orchestrations compared

| Stage | Common module (`tt_moe_decode.py`) | DeepSeek-V3 optimized (`moe_optimized.py`) |
|---|---|---|
| Router | caller supplies `scores`, `indices` | `MoEGate.forward` |
| Dispatch | `all_to_all_dispatch_metadata` | `all_to_all_dispatch_metadata` |
| Compute | `moe_compute` (Full, `optional_output_tensor`) | `moe_compute` (Full, `optional_output_tensor`) |
| Tilize | `deepseek_moe_post_combine_tilize` / `tilize_with_val_padding` | `deepseek_moe_post_combine_tilize` / `tilize_with_val_padding` |
| Score combine | `deepseek_moe_fast_reduce_nc_fused(scores, shared_expert_scale)` | `ttnn.mul(scores)` then `ttnn.sum` / `deepseek_moe_fast_reduce_nc` over K |
| Reduce-scatter | `deepseek_moe_reduce_scatter` \| `ttnn.reduce_scatter` \| skip | `deepseek_moe_reduce_scatter` \| `ttnn.reduce_scatter` |
| Prefill | (decode-shaped module) | `_fwd_prefill_moe`: batch-chunked, DRAM, seq↔batch interchange |

Both are the same shape: **dispatch → `moe_compute` (Full) → tilize → score-weighted reduce
over experts → reduce-scatter over the TP axis.**
