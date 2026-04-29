# TurboQuant KV Cache Quantization

## ✅ HYBRID COMBINE WORKING (2026-04-29 PM): W-sweep done

The sliding-window hybrid (TQ over `[0, cur_pos]` + BFP8 ring over recent W
tokens, combined via online-softmax LSE) now runs end-to-end with trace.

**N150 Llama-3.1-8B, 512-position teacher-forced eval, with trace:**

| Config | Top-1 | Top-5 | ms/tok | Δ top-1 vs Track A |
|---|---:|---:|---:|---:|
| Track A baseline (W=0) | 86.9 % | 97.1 % | 81.6 | — |
| Hybrid W=32  | 87.5 % | 97.3 % | 88.1 | +0.6 |
| Hybrid W=64  | 87.5 % | 97.5 % | 88.0 | +0.6 |
| **Hybrid W=128** | **88.9 %** | **97.9 %** | **88.1** | **+2.0** |
| Hybrid W=256 | 88.5 % | 97.7 % | 88.5 | +1.6 |

W=128 is the sweet spot: +2.0 pp top-1, +0.8 pp top-5 vs Track A at ~+6.5 ms/tok.
Memory overhead is negligible — 128 tokens × 8 KV heads × 128 dim × 1 byte × 32
layers ≈ 4 MB extra ring cache. Past W=128 the gain plateaus (and dips slightly
at W=256), so we don't need a bigger ring.

### Three fixes that unblocked this:

1. **Kernel: gate dequant by `!pre_rescaled`** instead of taking the
   `sdpa_standard` fast path (`sdpa_tq_decode.cpp`). The `pre_rescaled+
   return_lse` case now falls through to the existing Flash-Attention chunk
   loop, with Steps 0/1/4 (BFP8 norms typecast, K dequant, V dequant) gated
   by `if constexpr (!pre_rescaled)`. The post-loop LSE pack then runs for
   both modes. Without this the writer would deadlock on `cb_lse_out`.
2. **Combine: slice LSE col 0 before broadcasting to head_dim**
   (`ttnn_integration.py _combine_lse`). The LSE tile only has a meaningful
   value at column 0 — columns 1-31 are -inf/+inf/NaN from `log(0)` of the
   `matmul_reduce`'d rowsum and uninitialised cols of the max tile. The
   previous `ttnn.repeat([1,1,1,4])` propagated that garbage.  Slice→repeat
   guarantees all 128 cols hold col 0's value.
3. **Strategy: old SDPA gets full `current_pos` coverage, not `cur_pos-W`.**
   The TQ kernel rounds `valid_k_chunks` up to chunk boundary, so passing
   `cur_pos-W` produced ~31 positions of overlap with the ring — and the
   LSE-aware combine assumes disjoint coverage, so the result was 0 % top-1.
   Using full `current_pos` turns the combine into an "emphasis blend": the
   ring's BFP8-precision recent-token output gets up-weighted via LSE so
   recent tokens contribute more than their share. Beats Track A.

`old_pos` device tensor is still plumbed through `prepare_decode_inputs_host
→ hybrid_sdpa_decode` for future work; it is not used in the current strategy
but stays available for a clean disjoint-coverage variant if/when the kernel
grows per-token cur_pos masking.

### Ablation: per-head LSE weights at W=128

Set `TQ_DUMP_LSE=1` and run `eval_token_accuracy.py --tq-full-dequant
--tq-recent-window 128`. The hook in `_combine_lse` writes one block per
(step, layer<4) to `/tmp/tq_lse_dump.txt` with per-head `(LSE_old, LSE_new,
w_ring)`. Sample at layer 0, cur_pos≈600:

| Head | LSE_old | LSE_new | w_ring |
|---:|---:|---:|---:|
| 0  |  8.19 |  8.81 | **0.65** |
| 7  |  9.25 |  9.00 | 0.44 |
| 15 | 10.25 |  9.69 | 0.36 |
| 23 |  8.06 |  7.69 | 0.41 |
| 31 | 19.13 |  5.00 | **0.00** |

Heads specialise. Head 0 has roughly equal LSE on both halves but a slight
edge to ring (65% ring weight) — broad recent-context attention. Head 31 has
LSE_old=19 (a sharp distant peak — "needle"-style head looking back for a
specific far-away token) so the ring contributes essentially nothing. Average
over 1023 steps × 32 heads at layer 0 lands at ~15-22% ring weight, which
matches the +2 pp top-1 sweet spot at W=128: enough to nudge predictions
toward recent context without overpowering the long-range signal.

The combine is **not** a uniform 50/50 blend — it adaptively re-weights per
head and per step based on actual attention scores. That's why W=128 (88.9%)
beats the truncated-disjoint variant from earlier (which gave 0% top-1).

### Long-context smoke (`test_hybrid_long_context.py`)

Single-layer hybrid SDPA call at the end of a populated cache, 32 KV heads,
W=128, all on N150. Output finiteness + magnitude check only — no accuracy
metric (would need a longer reference text):

| Context | hybrid_sdpa_decode latency | Output max_abs | Result |
|---:|---:|---:|---|
| 4 K  | 4050 ms (first JIT) | 0.094 | PASS |
| 16 K | 2256 ms             | 0.044 | PASS |
| 32 K | 2233 ms             | 0.031 | PASS |
| 128 K | 2485 ms            | 0.029 | PASS |

The kernel scales to Llama's full 128 K window. Latencies are first-call
(JIT compile dominates ~2 s); steady-state per-call latency would be much
lower. No L1 overflow, no NaN propagation through the chunked online
softmax. Output magnitudes shrink at longer context as expected (more
positions to softmax over → narrower per-position weight).

Real per-token decode at 128 K would be 32× this (32 layers), but with
program-cache warm and trace capture the 2 s first-call dominates only
once.

### T3K validation: blocked on disk

`run_t3k_w_sweep.sh` ran with TT_NUM_DEVICES=8 but failed:
`TT_FATAL: Failed to write tensor data... No space left on device` while
sharding Llama-3.1-8B for T3K (ran out at layer 20/32 of `feed_forward.w1`).
`/localdev` is at 100% (3.3 T / 3.5 T). Big space hogs:

- 91 GB — `hf/ttnn_cache/meta-llama/Llama-3.3-70B-Instruct/`
- 17 GB — `hf/ttnn_cache/{N150,T3K}/` (orphan top-level, not used by current
  TT_CACHE_PATH)
- 16 GB — `hf/ttnn_cache/meta-llama/Llama-3.1-8B-Instruct/N150/` (the cache
  we used for the W-sweep)

T3K 8B cache needs ~25 GB. **Resolution requires user input** — either free
70B or orphan caches. Once unblocked, just re-run
`turbo_quant/run_t3k_w_sweep.sh`.

## 🚀 RESUME HERE — TurboQuant 128K plan complete (2026-04-29)

Status of the **TurboQuant 128K — Two-Track Comparison** plan:

| Plan task | Status |
|---|---|
| Track A: chunked fused TQ SDPA kernel | ✅ Already done before plan was written |
| Track B Step 1: synthetic BFP4 + std SDPA | ✅ `test_bfp4_paged_sdpa.py` cos ≥ 0.9995, 128 → 131072 |
| Track B Step 2: e2e `--tq-rescaled-bfp4` | ✅ N150 32-layer "Paris" answer; +15 % vs BFP8 |
| Track B Step 3: comparison sweep | 🟡 N150 32-layer measured; T3K + long context still TBD |
| Tier 2A E2E plumbing (`--tq-num-cores-per-head`) | ✅ Code in place, K=1 verified on N150 |

Headline finding: **Track B (`--tq-rescaled-bfp4`) is the simpler
winner.** It matches BFP8 baseline ms/tok within 15 % on N150 while
halving KV memory — no custom kernel, no chunked-streaming kernel
work needed, no cross-core orchestration.

### Two-track comparison

| Metric | Baseline BFP8 | Track A: TQ FD fused | Track B: TQ-rescaled BFP4 + std SDPA |
|---|---:|---:|---:|
| KV memory | 1 byte/elem | 0.5 byte/elem BFP4 idx + 0.5 norms padded | 0.5 byte/elem |
| SDPA latency | standard | fused TQ kernel | standard |
| Max context | 128 K | 128 K (chunked online softmax already in) | 128 K (validated, cos ≥ 0.9995) |
| Cosine vs CPU | — | > 0.999 (Tier 2A K = 14 PASS) | 0.9995 (synthetic) |
| E2E quality (N150 32-L "Paris") | "Paris" ✓ | not tested 32-L (would be ~250+ ms/tok at K=1) | "Paris" ✓ |
| E2E ms/tok (N150 32-L 256-tok) | 37.0 | not tested 32-L | 42.7 (+15 %) |
| Custom kernel | no | yes — sdpa_tq_decode.cpp + Tier 2A reducer | no — drops in over standard SDPA |
| Memory ratio vs BFP8 | 1.00× | ~0.75× (norms padding waste) | 0.50× |

### Open follow-ups (not blocking the plan)

1. **Long-context quality test for Track B** — extend prompt or
   `--max-new-tokens` to 4 K / 8 K / 16 K, confirm answer stays
   coherent.
2. **T3K K=14 e2e** — run with `--tq-full-dequant
   --tq-num-cores-per-head 14` on T3K, expect Tier 2A speedup to
   close the gap vs BFP8 baseline at long context.
3. **Final `bench_seqlen_sweep` row for `--tq-rescaled-bfp4`** —
   the existing sweep covers BFP8 and TQ FD; add a row for
   rescaled-BFP4 (it would just be paged BFP4 + standard SDPA, so
   ≈ BFP8 latency with 0.5× memory).

### Comparison snapshot (N150 K=1, cur_pos=seq-1, single SDPA call)

| seq    | TQ FD ms (fused) | BFP8 ms (std) | speedup |
|-------:|-----------------:|--------------:|--------:|
|   1024 |      2.25        |  0.06         |  0.03×  |
|   8192 |     17.69        |  0.15         |  0.01×  |
|  32768 |     70.65        |  0.41         |  0.01×  |
| 131072 |    282.37        |  1.45         |  0.01×  |

Per-call: BFP8 standard SDPA is ~195× faster than fused TQ at K=1
on N150. Tier 2A K=14 gives ~14× boost on T3K (so ~14× slower than
BFP8 at K=14). BFP4 paged + standard SDPA (Track B) should
match BFP8 latency while keeping 0.5× memory.

### Track B e2e validated (2026-04-29 N150 single device)

`--tq-rescaled-bfp4` quality + perf measured. Same prompt
("What is the capital of France?"), 32-layer Llama-3.1-8B,
30 generated tokens, traced execution:

| Mode                        | Answer                          | ms/tok | tok/s |
|-----------------------------|---------------------------------|-------:|------:|
| `--no-turbo-quant` (BFP8)   | "The capital of France is Paris." | 37.0  | 27.1  |
| `--tq-rescaled-bfp4`        | "The capital of France is Paris." | 42.7  | 23.4  |
| Δ                           | identical answer                | +5.7  | -3.7  |

Track B costs ~15 % e2e latency vs BFP8 baseline, gains 0.5×
KV memory. Quality preserved (correct, coherent answer; the
divergence at later tokens is the usual sampling sensitivity).

### T3K e2e three-way comparison (2026-04-29)

Same prompt + setup as N150 above, but on T3K (8 chips, KV heads
sharded 1 per device). Cache loaded from
`/localdev/mtairum/hf/ttnn_cache/T3K`:

| Mode                                    | Answer | ms/tok | vs BFP8   | KV memory |
|-----------------------------------------|:------:|-------:|----------:|----------:|
| `--no-turbo-quant` (BFP8 baseline)      | Paris  | **13.9** | 1.00×   | 1.00×     |
| `--tq-rescaled-bfp4` (Track B)          | Paris  | **18.8** | 1.35× slower | **0.50×** |
| `--tq-full-dequant --tq-num-cores-per-head 14` (Track A) | Paris | **29.1** | 2.09× slower | 0.75× |

**Verdict: Track B wins on every axis** — 1.55× faster than
Track A, halves the memory (vs Track A's 0.75×), and uses no
custom kernel. Track A's K=14 plumbing is verified live
(`[TQ Phase 2.3] num_cores_per_head req=14, max=16, clamped=14`),
but at cur_pos=72 SDPA is too small a fraction of step time for
the 13.7× per-call SDPA speedup to translate to e2e.

Track A's e2e gap to BFP8 is dominated by the rotated-path's
non-SDPA ops (matmul/norm work), not the SDPA cost — so K=14
parallelism cannot close it. Long-context (e.g. 32K+) might
narrow the relative gap somewhat as SDPA grows, but Track B
should hold its lead because BFP8 SDPA also stays fast there.

### Token-accuracy reality check (2026-04-29 N150)

`eval_token_accuracy.py` over 1024 reference tokens (eval = positions
512 → 1022, 511 positions scored):

| Mode                            | Top-1 | Top-5 | ms/tok | KV mem |
|---------------------------------|------:|------:|-------:|-------:|
| `--no-turbo-quant` (BFP8)       | **97.3%** | **100.0%** | 37.8 | 1.00× |
| `--tq-full-dequant` (Track A)   | **86.9%** | **97.1%**  | 83.1 | 0.75× |
| `--tq-rescaled-bfp4` (Track B)  | 66.6% | 82.2% | 43.5 | 0.50× |

**Both TQ paths lose accuracy vs BFP8.** Track A loses ~10 pp
top-1 / ~3 pp top-5; Track B loses ~31 pp / ~18 pp. Per-position
progression shows Track B steadily collapsing from ~84 % top-1
at pos 560 to ~67 % at pos 1010, while Track A holds ~84-92 %
across the same span — Track A errors are bounded, Track B's
errors compound.

So the short-context "Paris" tests at cur_pos = 72 didn't surface
either of these regressions; the synthetic per-call cosine
≥ 0.9995 on random K/V doesn't capture the real attention
dynamics either.

### Current recommendation

Neither track is yet a drop-in replacement for BFP8 baseline at
long context — both have meaningful quality regressions on this
benchmark. Track A is the better quality option (top-5 nearly
intact at 97.1 %), but is 2.2× slower than baseline on N150 K=1
and the regression in top-1 still matters for greedy decoding.

Both tracks stay in tree. The short-context win for Track B
(faster than BFP8, halves memory, identical "Paris" answer at
cur_pos ≤ 72) is real and useful for some workloads — but it
should not be defaulted on for general decoding without a
sliding-window hybrid or further work to recover the lost top-1.

### Track A gap — root cause (2026-04-29)

A/B test on N150, same 1024-position eval, default seed:

| Mode                                       | Top-1 | Top-5 |
|--------------------------------------------|------:|------:|
| BFP8 baseline (no TQ)                      | 97.3% | 100.0% |
| **TQ 3-bit + BFP8 paged + std SDPA** (default flags) | **86.1%** | **97.1%** |
| Track A (TQ 3-bit + BFP4 idx + fused kernel) | 86.9% | 97.1% |

**The Track A gap is the TQ 3-bit quantization itself, not the
fused kernel or BFP4 storage.** TQ rotation + 3-bit Lloyd-Max
quantization stored as BFP8 (the highest-precision config that
still uses TQ) lands at 86.1 % top-1 — within noise of Track A's
86.9 %. The fused kernel preserves the TQ baseline; it does not
add error.

Rotation absorption math verified at `ttnn_integration.py:1043`
(`wv_heads = rotation_t_cpu @ wv_heads` → Π^T·W_v) and `:1051`
(`wo_cols = wo_cols @ rotation_cpu` → W_o·Π). These compose
correctly: in rotated space attention_output is `attn·Π`, then
`(attn·Π)·W_o' = (attn·Π)·(W_o·Π)` — wait, that doesn't reduce.
Re-deriving: `attn·W_o^T_orig = (attn·Π)·(Π^T·W_o^T_orig) =
attn_rotated · (W_o·Π)^T = attn_rotated·W_o'^T` ✓. Math is right.

**The loss is purely from 3-bit quantization noise compounding
across 32 layers × ~10 K-chunks × 1024 positions.** Per-coord
3-bit Lloyd-Max on unit-Gaussian has ~14 dB SNR; tiny noise per
coord, but every K read adds it.

### `--bits 4` made it worse, not better — root cause found (2026-04-29)

Track A `--bits 4` on the same eval: **81.6 % top-1 / 93.6 % top-5**
— *worse* than 3-bit (86.9 % / 97.1 %).

**Root cause: BFP4 storage cannot represent the 4-bit index range.**
BFP4 has 1 sign bit + 3 mantissa bits per element with shared
8-bit exponent per 16 elements. For non-negative integer indices
0..N-1, only 8 distinct positive mantissas are representable per
exponent block.

Verified empirically (`/tmp/bfp4_indices_test.py`):
```
Test 1 (3-bit, indices 0-7):
  unique recovered: [0, 1, 2, 3, 4, 5, 6, 7]   max_diff = 0   ✓
Test 2 (4-bit, indices 0-15):
  unique recovered: [0, 2, 4, 6, 8, 10, 12, 14]  max_diff = 1  ✗
  row 0 original:   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15]
  row 0 recovered:  [0, 0, 2, 4, 4, 4, 6, 8, 8, 8,10,12,12,12,14,14]
Test 3 (4-bit, indices 0-15 via BFP8):
  unique recovered: [0, 1, ..., 15]   max_diff = 0   ✓
```

So `--bits 4` builds a 16-level Lloyd-Max codebook but the BFP4
cache erases half the distinctions; the kernel reads only 8
distinct indices and gathers WRONG centroid values for the lost
indices. This is strictly worse than 3-bit (which fits BFP4
exactly with no information loss).

**Implication for memory savings.** TQ's 0.75× memory ratio
relies on BFP4 indices (0.5 byte/coord). 4-bit TQ requires BFP8
storage (1 byte/coord) → memory parity with BFP8 baseline →
no advantage. So **3-bit is the only TQ configuration that
preserves the memory savings**. Higher-precision TQ would need
either:
  - non-BFP4 storage for indices (BFP8 / int8 / BF16) and
    accept memory parity, or
  - a different storage trick (e.g. pack two 4-bit indices into
    one BFP8 byte — but TTNN doesn't currently expose 4-bit int).

### TTNN-side precision points (2026-04-29 review)

Where else could precision hide in the pipeline? Quick survey:

1. **BFP4 index storage** — covered above. Hard limit at 3-bit.
2. **BF16 norms (TILE-padded)** — current. BF16 has ~3 decimal
   digits of precision; norm values are typically 5-15, fine.
   Switching to BFP8 norms saved memory (0.75× → ~0.50× ratio
   in earlier experiments) but added a ~0.5-1 pp accuracy cost.
   The current default is BF16 norms.
3. **Centroids in compute** — loaded as F32 compile-time
   constants, gathered via BF16 cascade in the SFPU. The cascade
   does N-1 BF16 add/multiply ops per coord; each accumulates
   tiny BF16 rounding. Could be tightened by keeping the gather
   in F32 until the final pack, but the gain is small for 8
   levels.
4. **Q × K^T** — BF16 inputs, F32 accumulator, BF16 output.
   Standard SDPA path. Not the bottleneck.
5. **Centroid × norm (Track B write path)** — BF16 × BF16 →
   BF16, then BF16 → BFP4. BFP4 has wide dynamic range but only
   3-bit mantissa. For typical centroid×norm values (e.g.
   1.95 × 10 ≈ 19.5), BFP4 rounds to ~16 or 20 — that's where
   Track B's −20 pp comes from on top of the TQ baseline.
6. **K rotation matmul** — BF16; Π is orthogonal so condition
   number is 1, no amplification of error.

Of these, the only one that matters for **Track A at 3 bits** is
#3 (compounding cascade error across 32 layers × many chunks
× 1024 positions). That's the irreducible cost of doing the
quantize/dequant inside the kernel at BF16. The fix would be
F32 internal accumulation in the cascade — non-trivial kernel
change.

### Tier 3A landed (2026-04-29) — +3.8 % perf, no quality cost

Hoisted all SFPU `*_init` calls in `dequant_k_chunk` and
`dequant_v_chunk` out of the per-tile and per-level loops. Init
macros configure the SFPU dispatcher; once configured, subsequent
ops of the same type don't need re-init. Drops ~480 init calls
per chunk to 6.

N150 bench delta:

| seq    | Pre-3A ms | Post-3A ms |  Δ    |
|-------:|----------:|-----------:|------:|
|   1024 |    2.23   |    2.16    | −3.1% |
|   8192 |   17.76   |   17.05    | −4.0% |
|  32768 |   70.68   |   67.96    | −3.8% |
| 131072 |  282.31   |  271.69    | −3.8% |

Correctness verified: 32-layer Llama-3.1-8B "Paris" answer
unchanged. Implementation in commit (this branch). Smaller than
the original tier estimate (5–10 %) but a clean, free win.

### Sliding-window hybrid: full LSE combine plumbed, validation pending (2026-04-29)

The proper LSE-aware combine is now plumbed end-to-end. Smoke passes
("Paris" at 99.3 ms/tok with W=64). 1024-position accuracy run blocked
on a perf issue, not correctness — see "Open issues" below.

**Implementation summary** (commits `9edefa7c385`, `e36f336d278`,
`8227bc32354`):

- **Step 1 — Fused TQ kernel LSE export.** New `return_lse` compile-time
  flag; before final divide, compute `LSE = max·scale + log(sum)` and
  pack to `cb_lse_out` (c_3, repurposed from Tier 2A's
  `cb_merge_new_max`). Mutually exclusive with `num_cores_per_head > 1`.
- **Step 2 — Plumbing.** Writer reads cb_lse_out and writes to a second
  output tensor `[B, NQH, 1, 32]` BF16 (1 tile per head, broadcast
  across the tile from the row-sum reduce). Device op now returns
  `std::vector<Tensor>` of size 1 or 2. Python prim + nanobind +
  `ttnn_integration.fused_sdpa_decode` accept `return_lse=True` and
  return `(out, lse)` tuple.
- **Step 3 — pre_rescaled trick.** Skipped modifying the standard SDPA
  decode kernel by reusing the fused TQ kernel's existing pre_rescaled
  mode for the BFP8 ring half (bypasses dequant cascade, runs
  sdpa_standard with BFP8 K/V typecast inline). LSE export from step 1
  already works there. Saved ~1 day of shared-kernel surgery.
- **Step 4 — Host combine.** New `TTNNTurboQuantCache.hybrid_sdpa_decode`
  runs both halves and combines via the standard online softmax LSE
  formula on device:
  ```
  lse_max  = max(lse_old, lse_new)
  w_X      = exp(lse_X − lse_max) / (exp(lse_old − lse_max) + exp(lse_new − lse_max))
  out      = out_old · w_old + out_new · w_new
  ```
  LSE tensors are `[B,NQH,1,32]`; out is `[B,NQH,1,128]`. Used
  `ttnn.repeat([1,1,1,4])` to expand LSE columns 32→128 for the multiply.
- attention.py routes through `hybrid_sdpa_decode` when
  `recent_window > 0`. eval_token_accuracy.py disables trace capture
  when sliding-window mode is on (the dual-write path needs a host read
  of cur_pos to compute `cur_pos % W`).

**Verification:** `test_lse_export.py` PASS (out bit-identical with
return_lse=False; LSE values 3.83-4.66 at cur_pos=42 — matches
log(42)+max·scale ≈ 3.7-5). 32-layer "Paris" e2e at W=64 produces
correct answer at 99.3 ms/tok.

**Open issues for next session:**

1. **1024-position accuracy still slow even after host-sync 32× cut.**
   Tried caching cur_pos %W at layer_idx=0 (saves 32× host syncs per
   step). Did not fix the bottleneck — the test still sat in setup
   after 2-3 minutes. Likely cause: the no-trace decode path has
   per-op Python/dispatch overhead that dominates with the doubled
   SDPA call count + 5 extra ttnn ops in the LSE combine. Or the
   first-N-step JIT compile is much heavier than expected because the
   hybrid path creates two distinct programs per step.

   **Fix paths to try, in order:**
   (a) Profile the no-trace per-step time with smaller eval
       (`--num-eval-tokens 50` is currently a no-op since 1023 forward
       passes still happen — would need a real fast-exit). Goal:
       confirm whether bottleneck is JIT vs dispatch.
   (b) Plumb `ring_pos` as a separate trace-input tensor (the original
       proper fix) — reenables trace and cuts dispatch overhead 10-50×.
   (c) Drop the on-device LSE combine in favour of host-side combine
       (small tensor reads, simple math, 1 round-trip per layer).
       Probably faster than the current 5-op device path.

2. **W-sweep validation** (W ∈ {32, 64, 128, 256}) — runs once #1 lands.
3. **Standard SDPA decode kernel LSE export** — no longer needed thanks
   to the pre_rescaled trick. Drop from the design doc.



`turbo_quant/SLIDING_WINDOW_DESIGN.md` has the full design.
Implementation status (commit `3dcc01b1db9`):

- **Step 1 cache structure** ✅ — BFP8 ring + cyclic page table.
- **Step 2 dual-write** ✅ — `update_cache` writes K/V to both
  the TQ cache and the BFP8 ring. Two real implementation snags
  found and worked around (quantize() aliases its input, paged
  update_cache won't accept page_table > cache size).
- **Step 3 hybrid SDPA** 🟡 — first cut shipped as **ring-only**
  (standard SDPA over the BFP8 ring, ignoring TQ contributions
  from older positions). Tested the cheaper hypothesis first.
  **Full hybrid with online-softmax combine still TODO** — needs
  the standard SDPA decode kernel to expose LSE so we can
  log-sum-exp two SDPA outputs at host level.
- **Step 4 eval flags** ✅ — `--tq-recent-window N`.
- **Step 5 validation** ✅ for ring-only (see below).

### Ring-only validation result (2026-04-29 N150 W=64)

`eval_token_accuracy.py --tq-full-dequant --tq-recent-window 64`:

| Mode                            | Top-1 | Top-5 |
|---------------------------------|------:|------:|
| BFP8 baseline                   | 97.3% | 100.0% |
| Track A (TQ FD K=1)             | 86.9% | 97.1%  |
| Track B (TQ rescaled BFP4)      | 66.6% | 82.2%  |
| **SW ring-only W=64**           | **24.0%** | **36.7%** |

**Hypothesis "Llama tolerates W=64 sliding-window attention"
→ REJECTED.** Per-position trace is stable (24-43 %, no monotonic
decay) — just inherently low because the model needs long-range
context. Discarding everything past W is too aggressive.

So the **full hybrid (recent BFP8 + old TQ + LSE-based online
softmax combine) is the only path that can recover quality**.
The ring infrastructure is still useful — it provides the
high-precision recent K/V the combine math needs — but the
combine step is mandatory, not optional.

Required follow-up for the combine:
- Modify `paged_scaled_dot_product_attention_decode` to optionally
  return LSE = log(sum) + max alongside the output. The kernel
  already maintains (max, sum) internally for online softmax.
- Modify the fused TQ SDPA kernel similarly.
- Host-level combine: `out = (out_a · exp(lse_a − lse_max) +
  out_b · exp(lse_b − lse_max)) / (exp(lse_a − lse_max) +
  exp(lse_b − lse_max))`.

Estimated additional effort: ~1-2 days for the LSE-aware kernel
+ host combine.

### Open follow-ups

- **Sliding-window hybrid implementation** (per design doc) —
  the actual fix for Track A's quality gap.
- **T3K K=14 + Track A accuracy** — same 1024-position run on
  T3K should give the same top-1/top-5 (numerics deterministic)
  at much lower latency.
- **Add `--tq-full-dequant` row** to `bench_seqlen_sweep.py`
  with K=14 numbers when run on T3K.

### N150 per-call SDPA latency sweep (2026-04-29, K=1)

Three-way per-call SDPA latency at `cur_pos = seq - 1`
(`bench_seqlen_sweep.py` now reports all three columns):

| seq    | TQ FD ms | BFP4 ms | BFP8 ms | TQ/BFP8 | BFP4/BFP8 |
|-------:|---------:|--------:|--------:|--------:|----------:|
|    128 |   0.30   |  0.04   |  0.04   |  0.75×  |   0.50×   |
|   1024 |   2.23   |  0.05   |  0.06   |  0.75×  |   0.50×   |
|   8192 |  17.76   |  0.12   |  0.15   |  0.75×  |   0.50×   |
|  16384 |  35.37   |  0.18   |  0.24   |  0.75×  |   0.50×   |
|  32768 |  70.68   |  0.30   |  0.42   |  0.75×  |   0.50×   |
|  65536 | 141.24   |  0.53   |  0.78   |  0.75×  |   0.50×   |
| 131072 | 282.31   |  0.92   |  1.46   |  0.75×  |   0.50×   |

**Track B (BFP4) is actually faster than BFP8** at long context
(the SDPA decode op is memory-bound — half the bytes to read
beats slightly slower compute). At 131 K: BFP4 = 0.92 ms vs
BFP8 = 1.46 ms (1.59× faster) vs Track A K=1 = 282 ms (~307×
faster than Track A).

### Tier 2A — DONE (commits e74354b → e6aa928)

Phase 2.3 (cross-core split) landed end-to-end on T3K. Full bench
(`bench_seqlen_sweep.py`, `cur_pos = 1791` across all seqs):

| seq    | K=1 ms | K=14 ms | speedup |
|-------:|-------:|--------:|--------:|
| 1 024  |  2.24  |  0.37   |  6.05×  |
| 2 048  |  4.43  |  0.64   |  6.92×  |
| 4 096  |  8.84  |  0.92   |  9.61×  |
| 8 192  | 17.71  |  1.51   | 11.7×   |
| 16 384 | 35.38  |  2.88   | 12.3×   |
| 32 768 | 70.57  |  5.37   | 13.1×   |
| 65 536 | 141.11 | 10.35   | 13.6×   |
| 131 072| 282.07 | 20.56   | 13.7×   |

Approaches the theoretical 14× as work-per-core dominates the merge
cost. Cosine vs masked reference: K = 1, 2, 4, 8, 14 all PASS at
cur_pos ∈ {41, 200, 1791} (empty-slice fast path verified for the
12-13 idle workers at small cur_pos).

**Empty-slice fix (Phase 2.3 step 5, commit e6aa928bd27)**: workers
whose `[chunk_start, chunk_end)` range is empty push neutral tiles
(max = -10000.0f, sum = 0, out = 0) to cb_partial_* and sema_inc
as usual, so the reducer's merge becomes a no-op for that peer.
This is what unlocks the small-cur_pos rows of the table above.

### Tier 2A — deferred follow-ups (not blocking 128K plan)

- **E2E with K=14** — plumb `num_cores_per_head` through
  `models/tt_transformers/tt/attention.py` `tq_cache.fused_sdpa_decode`
  call + CLI flag in `eval_e2e.py`. The fused kernel is already
  generic; just needs the host wiring. Table speedups are per-call;
  e2e with K=14 should land close to the BFP8 baseline at long context.
- **N150 single-device** stays at K=1 (max_cores_per_head =
  num_cores / (B · NQH) = 56/32 = 1). Tier 2A is a multi-device
  win only.

### Files touched in Phase 2.3 (commits a6d2903 → 51c7ed3)

- `ttnn/cpp/.../sdpa/device/sdpa_tq_device_operation.{cpp,hpp}` — `num_cores_per_head` attribute
- `ttnn/cpp/.../sdpa/device/sdpa_tq_program_factory.cpp` — work distribution + multi-slot CBs + per-core args + reducer NoC coords + semaphore
- `ttnn/cpp/.../sdpa/kernels/compute/sdpa_tq_decode.cpp` — chunk-slice math, worker pack-and-skip, reducer wait-and-merge
- `ttnn/cpp/.../sdpa/kernels/dataflow/reader_tq_decode.cpp` — chunk-slice math
- `ttnn/cpp/.../sdpa/kernels/dataflow/writer_tq_decode.cpp` — worker NoC-send + reducer wait + cb_push_back
- `ttnn/cpp/.../turbo_quant/turbo_quant.{cpp,hpp}` — add num_cores_per_head kwarg
- `ttnn/cpp/.../turbo_quant/turbo_quant_nanobind.cpp` — pybind kwarg
- `turbo_quant/ttnn_integration.py` — `fused_sdpa_decode` accepts num_cores_per_head
- `turbo_quant/test_2A_cores_per_head.py` — N150 sweep (always clamps to K=1)
- `turbo_quant/test_mesh_fused_sdpa.py` — T3K mesh test reads `TQ_NUM_CORES_PER_HEAD` env var
- `turbo_quant/bench_seqlen_sweep.py` — same env var
- `turbo_quant/TIER_2A_DESIGN.md` — full design + status notes for each step

---

## ✅ E2E QUALITY FIXED (2026-04-28 PM)

Both root-cause bugs are now patched. End-to-end output at 32 layers:

```
Q: What is the capital of France?
A: The capital of France is Paris.
```

**Fix 1 — V double-rotation in `update_cache`** (`turbo_quant/ttnn_integration.py`):
When `rotation_absorbed=True`, V already comes pre-rotated from W_v (the
absorbed-rotation state dict applies Π^T to W_v so V_output = V @ Π). The
old `update_cache` then called `quantize(v_heads)` without `skip_rotation`,
applying Π again → V was double-rotated (V @ Π²) before being stored.
Performance mode already passed `skip_rotation=rotation_absorbed`; the fused-
SDPA path was missing this. Symptom: 1-layer logits magnitude ~40 % too
large (10.13 vs 7.25 baseline) before fix.

**Fix 2 — softmax-denominator dilution correction** (`sdpa_tq_decode.cpp`):
The fused kernel iterates a full 128-token chunk regardless of `cur_pos`.
Standard SDPA decode masks score positions `> cur_pos` to NEG_INF so they
contribute 0 to softmax. The TQ kernel did not — zero-K positions instead
contribute `exp(-max * scale)` to the denominator each, diluting real
attention weights. After matmul_reduce of the running sum, we now subtract
`zero_count * exp(-max * scale)` from `prev_sum` before reciprocating. This
is mathematically equivalent to NEG_INF masking because V[zero] = 0 already
zeroes the numerator contribution.

Validation:
- `turbo_quant/test_partial_cache.py` (synthetic, GQA): cos > 0.9996 vs
  masked reference at every cur_pos; ratio ~0.97-1.02.
- `turbo_quant/test_paged_partial_cache.py` (e2e-mimicking, scatters via
  paged_update_cache): cos = 0.997, ratio = 0.97.
- 32-layer e2e: max=32.5 (matches baseline 33.5), top-1 = "The", coherent
  multi-token output.

## Mesh + seqlen sweep (2026-04-28, post-fix)

**3B mesh validation:** `test_mesh_fused_sdpa.py` on T3K, all 8 devices
PASS (cos 0.995-0.998 vs masked ref, ratio 0.96-1.00).

**3C T3K e2e:** `eval_e2e.py --tq-full-dequant TT_NUM_DEVICES=8` traced =
**32.6 ms/tok (30.6 tok/s)**, output "The capital of France is Paris."

### Seqlen sweep — `bench_seqlen_sweep.py`

Per-call SDPA decode latency at `cur_pos = seq - 1`, 32-layer Llama-3.1-8B
(8 KV heads / 32 Q heads / head_dim=128). KV totals are summed across
all 32 layers and all devices.

**N150 (1 chip):**

| seq | TQ FD ms | BFP8 ms | speedup | TQ KV total | BFP8 total | TQ/BFP8 | TQ idx (BFP4) | TQ norms (BF16) |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 128 | 0.48 | 0.04 | 0.09× | 8 MB | 8 MB | 1.00× | 4 MB | 4 MB |
| 256 | 0.94 | 0.04 | 0.05× | 16 MB | 16 MB | 1.00× | 8 MB | 8 MB |
| 512 | 1.84 | 0.05 | 0.03× | 32 MB | 32 MB | 1.00× | 16 MB | 16 MB |
| 1 024 | 3.66 | 0.06 | 0.02× | 64 MB | 64 MB | 1.00× | 32 MB | 32 MB |
| 2 048 | 7.29 | 0.07 | 0.01× | 128 MB | 128 MB | 1.00× | 64 MB | 64 MB |
| 4 096 | 14.56 | 0.09 | 0.01× | 256 MB | 256 MB | 1.00× | 128 MB | 128 MB |
| 8 192 | 29.08 | 0.16 | 0.01× | 512 MB | 512 MB | 1.00× | 256 MB | 256 MB |
| 16 384 | 58.11 | 0.24 | 0.00× | 1.00 GB | 1.00 GB | 1.00× | 512 MB | 512 MB |
| 32 768 | 116.16 | 0.42 | 0.00× | 2.00 GB | 2.00 GB | 1.00× | 1.00 GB | 1.00 GB |
| 65 536 | 232.30 | 0.78 | 0.00× | 4.00 GB | 4.00 GB | 1.00× | 2.00 GB | 2.00 GB |
| 131 072 | 464.37 | 1.47 | 0.00× | 8.00 GB | 8.00 GB | 1.00× | 4.00 GB | 4.00 GB |

**T3K (8 chips, KV heads sharded 1 per device):**

| seq | TQ FD ms | BFP8 ms | speedup | TQ KV total | BFP8 total | TQ/BFP8 | TQ idx (BFP4) | TQ norms (BF16) |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 128 | 0.48 | 0.10 | 0.21× | 8 MB | 8 MB | 1.00× | 4 MB | 4 MB |
| 256 | 0.94 | 0.16 | 0.17× | 16 MB | 16 MB | 1.00× | 8 MB | 8 MB |
| 512 | 1.85 | 0.11 | 0.06× | 32 MB | 32 MB | 1.00× | 16 MB | 16 MB |
| 1 024 | 3.66 | 0.11 | 0.03× | 64 MB | 64 MB | 1.00× | 32 MB | 32 MB |
| 2 048 | 7.28 | 0.11 | 0.02× | 128 MB | 128 MB | 1.00× | 64 MB | 64 MB |
| 4 096 | 14.55 | 0.12 | 0.01× | 256 MB | 256 MB | 1.00× | 128 MB | 128 MB |
| 8 192 | 29.08 | 0.15 | 0.01× | 512 MB | 512 MB | 1.00× | 256 MB | 256 MB |
| 16 384 | 58.08 | 0.15 | 0.00× | 1.00 GB | 1.00 GB | 1.00× | 512 MB | 512 MB |
| 32 768 | 116.09 | 0.15 | 0.00× | 2.00 GB | 2.00 GB | 1.00× | 1.00 GB | 1.00 GB |
| 65 536 | 232.16 | 0.24 | 0.00× | 4.00 GB | 4.00 GB | 1.00× | 2.00 GB | 2.00 GB |
| 131 072 | 464.15 | 0.33 | 0.00× | 8.00 GB | 8.00 GB | 1.00× | 4.00 GB | 4.00 GB |

### Two big findings

**1. KV cache savings are 0%** — TQ FD is the same size as BFP8. The BF16
norms tensor is shaped `[max_blocks, num_kv_heads, block_size, 1]` and
`TILE_LAYOUT` pads `[block_size, 1]` to `[32, 32]`, so 31/32 of the norms
storage is wasted padding. That waste is exactly the size of the BFP4
indices, so the savings cancel out at every seqlen. **PARTIAL FIX → BFP8
norms now lands the ratio at 0.75×, see below.**

**2. TQ FD per-call latency is 5-100× slower than BFP8** and scales linearly
with `cur_pos` (~0.48 ms / 128 tokens per layer). T3K does not help — each
device runs the full chunk loop on its local head shard. BFP8 stays
sub-millisecond up to 128K. At long context this gap dominates e2e.

## Performance optimization plan & status (2026-04-28 PM)

### Where we are vs. BFP8 baseline (T3K traced, 42-tok prompt)

| Mode | T3K ms/tok | vs BFP8 | KV memory |
|---|---:|---:|---:|
| Baseline BFP8 | 13.9 | 1.00× | 1.00× |
| **TQ Pre-Rescaled BFP8 (Tier 1C)** | **18.8** | **1.35×** | 1.00× |
| TQ Full Dequant (post-cascade fix) | 26.9 | 1.94× | 0.75× |
| TQ Full Dequant (pre-cascade fix) | 32.7 | 2.35× | 0.75× |

### Plan tiers (all keep TQ rotation + accuracy; differ on memory vs latency)

**Tier 1 — eliminate dequant cost**
- 1A: Fused dequant-into-matmul (custom LLK reading BFP4+norms inline). ~1-2 wk. Keeps 0.75× memory, eliminates ~60% of per-chunk cost.
- 1B: BFP8 indices with codebook values directly. Loses BFP4 advantage (1.0× memory).
- **1C: Pre-rescaled BFP8 cache + standard SDPA. ✅ DONE. 1.35× of baseline, 1.0× memory.**

**Tier 2 — parallelise the chunk loop**
- 2A: Cross-core chunk distribution + ring-reduce online softmax. ~2 wk. Keeps 0.75× memory. **Linear speedup with cores per (B, NQH) tuple — biggest long-context lever.**
- 2B: Pack multiple Q-heads onto fewer cores. Marginal.

**Tier 3 — incremental per-chunk trims**
- 3A: Hoist cascade `*_init` calls outside per-tile loop. ½ day. 5-10% off cascade.
- 3B: Drop the BFP4→BF16 typecast pre-pass (run cascade ops on BFP4-input directly). 1-2 days.
- 3C: Skip BFP8-norms typecast (LLK fix). Defer.

**Tier 4 — algorithmic**
- 4A: Sliding-window hybrid (recent tokens BFP8, old tokens TQ). ~1-2 wk. Changes memory profile.
- 4B: TopK / sparse attention. Research.

### Decision matrix — which option keeps memory + most perf?

For preserving the 0.75× memory advantage of TQ FD:

| Option | At 32K, 32 layers | Effort | Risk |
|---|---|---|---|
| Today | 70 ms/call (TQ FD) | — | — |
| 1A | ~25 ms/call (-65%) | 1-2 wk LLK | high (custom matmul) |
| **2A** | **~9 ms/call (-87%)** | **~2 wk** | **medium (known pattern)** |
| 1A + 2A | ~3 ms/call (-96%) | 3-4 wk | combined |

**Going with 2A first** — it dominates at long context (which is the actual blocker), follows the existing `sdpa_flash_decode` cross-core template, and compounds with 1A later if needed.

### Recommended execution order

1. **2A — cross-core chunk distribution** ← starting now.
2. 3A + 3B (½-1 day each, cheap follow-ups)
3. 1A only if 2A doesn't close the gap enough at 128K.
4. 4A only as a fallback for extreme context (>128K).

## Centroid-gather cascade optimization (2026-04-28 PM)

The TQ chunk-loop bottleneck was the per-tile centroid-gather telescoping
cascade — 6 SFPU ops per level × 7 levels × 32 K/V tiles per chunk. Two
redundancies eliminated:

1. **DST→DST copy instead of CB→DST**. The cascade reloaded `idx` from
   `cb_dq_temp` every level even though it's constant; now loaded once
   into DST 0 and refreshed via `copy_dest_values<BF16>(0, 2)`.
2. **Precomputed centroid deltas + `mul_unary_tile`**. Replaced runtime
   `fill_tile + sub_binary_tile + mul_binary_tile` with a single
   `mul_unary_tile(2, delta_bits[lev])` against deltas computed once at
   kernel_main start.

Per-level: 6 ops → 4 ops (-33%). Per-chunk: ~40% fewer SFPU ops.

### Updated seqlen sweep — post-cascade-optimization

**N150 (1 chip):**

| seq | TQ FD ms | Δ vs prev | BFP8 ms | KV ratio |
|--:|--:|--:|--:|--:|
| 128 | 0.30 | -38% | 0.04 | 0.75× |
| 256 | 0.57 | -39% | 0.04 | 0.75× |
| 512 | 1.12 | -39% | 0.05 | 0.75× |
| 1 024 | 2.23 | -39% | 0.06 | 0.75× |
| 2 048 | 4.42 | -40% | 0.07 | 0.75× |
| 4 096 | 8.82 | -40% | 0.10 | 0.75× |
| 8 192 | 17.62 | -40% | 0.16 | 0.75× |
| 16 384 | 35.20 | -40% | 0.25 | 0.75× |
| 32 768 | 70.30 | -40% | 0.43 | 0.75× |
| 65 536 | 140.59 | -40% | 0.78 | 0.75× |
| 131 072 | 281.02 | -40% | 1.47 | 0.75× |

**T3K (8 chips):**

| seq | TQ FD ms | Δ vs prev | BFP8 ms |
|--:|--:|--:|--:|
| 128 | 0.30 | -38% | 0.10 |
| 256 | 0.57 | -39% | 0.16 |
| 512 | 1.11 | -40% | 0.11 |
| 1 024 | 2.22 | -39% | 0.12 |
| 2 048 | 4.41 | -40% | 0.11 |
| 4 096 | 8.80 | -40% | 0.14 |
| 8 192 | 17.62 | -40% | 0.17 |
| 16 384 | 35.13 | -40% | 0.14 |
| 32 768 | 70.20 | -40% | 0.16 |
| 65 536 | 140.36 | -40% | 0.29 |
| 131 072 | 280.56 | -40% | 0.34 |

**E2E:** T3K traced `--tq-full-dequant` 32 layers: 32.7 → 26.9 ms/tok
(**18% faster end-to-end**), output "The capital of France is Paris."
unchanged.

**Validation:** cos > 0.995 on N150 partial cache + all 8 T3K devices.

## BFP8 norms repack (2026-04-28 PM)

Cut on-device norms storage in half by storing as BFP8_B in DRAM and
typecasting to BF16 in the compute kernel before `mul_tiles_bcast_cols`
(which does not natively unpack BFP8 input).

**Wiring:** TTNNTurboQuantCache now allocates norms as `bfloat8_b`. The
device op accepts BF16 or BFP8 norms. The program factory adds two BF16
scratch CBs (`c_15` for K, `c_17` for V) and passes a `norms_are_bfp8`
compile-time flag. The compute kernel runs a per-chunk typecast helper
before each `dequant_*_chunk` when the flag is set; `cb_k_norms` /
`cb_v_norms` aliases route the dequant body at the BF16 scratch.

### Updated seqlen sweep — BFP8 norms

**N150 (1 chip):**

| seq | TQ FD ms | BFP8 ms | speedup | TQ KV total | BFP8 total | TQ/BFP8 | TQ idx (BFP4) | TQ norms (BFP8) |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 128 | 0.48 | 0.04 | 0.09× | 6 MB | 8 MB | **0.75×** | 4 MB | 2 MB |
| 256 | 0.94 | 0.04 | 0.05× | 12 MB | 16 MB | **0.75×** | 8 MB | 4 MB |
| 512 | 1.85 | 0.05 | 0.03× | 24 MB | 32 MB | **0.75×** | 16 MB | 8 MB |
| 1 024 | 3.68 | 0.06 | 0.02× | 48 MB | 64 MB | **0.75×** | 32 MB | 16 MB |
| 2 048 | 7.36 | 0.07 | 0.01× | 96 MB | 128 MB | **0.75×** | 64 MB | 32 MB |
| 4 096 | 14.64 | 0.10 | 0.01× | 192 MB | 256 MB | **0.75×** | 128 MB | 64 MB |
| 8 192 | 29.31 | 0.16 | 0.01× | 384 MB | 512 MB | **0.75×** | 256 MB | 128 MB |
| 16 384 | 58.52 | 0.24 | 0.00× | 768 MB | 1.00 GB | **0.75×** | 512 MB | 256 MB |
| 32 768 | 116.97 | 0.41 | 0.00× | 1.50 GB | 2.00 GB | **0.75×** | 1.00 GB | 512 MB |
| 65 536 | 233.86 | 0.77 | 0.00× | 3.00 GB | 4.00 GB | **0.75×** | 2.00 GB | 1.00 GB |
| 131 072 | 468.87 | 1.47 | 0.00× | 6.00 GB | 8.00 GB | **0.75×** | 4.00 GB | 2.00 GB |

**T3K (8 chips):**

| seq | TQ FD ms | BFP8 ms | speedup | TQ KV total | BFP8 total | TQ/BFP8 | TQ idx (BFP4) | TQ norms (BFP8) |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 128 | 0.48 | 0.10 | 0.21× | 6 MB | 8 MB | **0.75×** | 4 MB | 2 MB |
| 256 | 0.94 | 0.16 | 0.17× | 12 MB | 16 MB | **0.75×** | 8 MB | 4 MB |
| 512 | 1.85 | 0.11 | 0.06× | 24 MB | 32 MB | **0.75×** | 16 MB | 8 MB |
| 1 024 | 3.68 | 0.12 | 0.03× | 48 MB | 64 MB | **0.75×** | 32 MB | 16 MB |
| 2 048 | 7.33 | 0.11 | 0.01× | 96 MB | 128 MB | **0.75×** | 64 MB | 32 MB |
| 4 096 | 14.64 | 0.11 | 0.01× | 192 MB | 256 MB | **0.75×** | 128 MB | 64 MB |
| 8 192 | 29.28 | 0.14 | 0.00× | 384 MB | 512 MB | **0.75×** | 256 MB | 128 MB |
| 16 384 | 58.51 | 0.14 | 0.00× | 768 MB | 1.00 GB | **0.75×** | 512 MB | 256 MB |
| 32 768 | 116.88 | 0.15 | 0.00× | 1.50 GB | 2.00 GB | **0.75×** | 1.00 GB | 512 MB |
| 65 536 | 233.66 | 0.23 | 0.00× | 3.00 GB | 4.00 GB | **0.75×** | 2.00 GB | 1.00 GB |
| 131 072 | 467.13 | 0.33 | 0.00× | 6.00 GB | 8.00 GB | **0.75×** | 4.00 GB | 2.00 GB |

**Validation:**
- `test_paged_partial_cache.py`: cos = 0.997 (unchanged)
- `test_mesh_fused_sdpa.py` on T3K: all 8 devices PASS (cos 0.996-0.998)
- N150 e2e 32 layers: "The capital of France is Paris." (correct)
- T3K e2e 32 layers traced: 32.7 ms/tok (vs 32.6 ms/tok pre-change → typecast
  pass adds ~0.3% overhead, in noise)

**To get below 0.75× ratio**, the norms tensor still has 31/32 of its tile
cols zero-padded — repacking 32 blocks per tile (instead of 1) would cut
norms by another ~32× and bring the total ratio under 0.5×. That requires
bypassing `paged_update_cache` for norms (it writes whole rows and assumes
1 block per tile-row), which is a deeper refactor.

## Earlier (2026-04-27 evening) — perf bottleneck

**Headline:** the fused-SDPA kernel was iterating ALL 256 padded k_chunks
(1024 max_blocks / Sk_chunk_t=4) regardless of `cur_pos`. For seq=128 only
1 chunk has real data — we were doing 256× more compute than needed. Fixed
by plumbing `cur_pos` into the kernel and bounding the loop dynamically
(commit `d7f7585a9da`).

**Trace numbers (N150, --tq-full-dequant, 2026-04-27 evening):**

| Layers | Before (ms/tok) | After (ms/tok) | Speedup |
|-:|-:|-:|-:|
| 2 | 237 | **6.0** | 39× |
| 32 | ~3700 | **58.1** | 64× |

vs baseline BFP8 (37 ms/tok), Full Dequant is now **1.6× from baseline** at
32L instead of 100×.

### Open issue: e2e output quality is wrong (kernel is correct in isolation)

The kernel itself is correct in isolation — verified against multiple
torch references:

| Test | Config | cos vs torch | Verdict |
|---|---|---:|---|
| `test_paged_fused.py` | seq=128–2048, full-fill, NKH=8/NQH=8 | 0.9996+ | PASS |
| `test_padded_cache.py` | real=42, max_blocks=1024, cap=1 | 0.9939 | PASS |
| `test_e2e_writes.py` | + paged_update_cache writes | 0.9996 | PASS |
| `test_e2e_writes.py` (with GQA + pre-rotated Q) | NQH=32/NKH=8, Q × Π | 0.9996 | PASS |
| `test_bfp4_roundtrip.py` | BFP4 lossless for integers 0–7 | exact | PASS |

But end-to-end generation (eval_e2e.py 32L) produces garbage:

| Variant | 32L "What is the capital of France?" |
|---|---|
| Baseline BFP8 (`--no-turbo-quant`) | "The capital of France is Paris" ✓ |
| BFP4 paged (`--bfp4-cache`, std SDPA) | "The capital of France is Paris" ✓ |
| TQ Full Dequant, no cap | " ( the first, the" — broken English |
| TQ Full Dequant, cap=1 (HEAD) | "OOOOak\nOOak" — random |
| TQ Full Dequant, cap=2 (forced min) | "OOOOOOOOOOOO" — collapse |
| TQ Full Dequant 1L | "ONUS .bz #..." — random (1L baseline also random) |

Even with `--no-trace`, `--num-layers 1`, and at single-device, the bug
persists. The pre-rescaled BFP4 path WORKS — same TQ quantization, just
stored as `centroid_value × norm` and read by std SDPA. The fused kernel
which stores `index + norm` separately and dequantizes inside, doesn't.

**Confidence note:** unit tests prove the kernel computes
`Q × dequant(idx, norm)^T × dequant(idx, norm) = Q × K^T × V` correctly
even with paged + GQA + pre-rotated Q + 1024-block padding + cap=1.
But somewhere between the unit test setup and eval_e2e.py's setup,
something diverges.

Bisect candidates ruled out (in two debug sessions):
- trace mode (`--no-trace` also fails)
- BFP4 quantization of integer indices 0–7 (lossless verified by
  `test_bfp4_roundtrip.py`)
- `paged_update_cache` write path (`test_e2e_writes.py` passes)
- GQA grouping (`test_e2e_writes.py` with NQH=32/NKH=8 passes)
- pre-rotated Q (`test_e2e_writes.py` with `Q × Π` passes)
- partial-fill softmax dilution (`test_padded_cache.py` with cap=1 passes
  cos > 0.99 even with 1020 blocks of zero-K padding)
- centroid-list mismatch host vs kernel (both use `get_codebook(...).centroids`)
- rotation seed mismatch (eval_e2e uses seed=42, TTNNTurboQuantSetup
  default is also seed=42)
- **kernel reads correct `cur_pos`** (DPRINT in e2e shows 0,1,2,…,44)
- **kernel reads correct Q** (DPRINT shows real BF16 floats like
  `0.2461 -0.7031 0.2197 -0.9258`, not integer-looking quantized values)
- **kernel reads correct K_idx in paged cache** (DPRINT confirms at
  cur_pos=N, rows 0..N have real data and persist across steps; row 0
  written at step 0 stays the same through step 41)
- **kernel reads correct K_norms** (DPRINT shows distinct per-position
  values 11.6250, 1.1328, 15.3125, 0.7617, 20.1250, 15.5000, 14.4375,
  13.7500 — varied as expected, persist across steps)
- **kernel reads correct V_idx and V_norms** (DPRINT shows valid
  integer-valued V_idx 0..7 and reasonable V_norms 0.04–0.43)
- post-SDPA permute axis ordering (`(2, 1, 0, 3)` to match std SDPA
  shape broke sharding; `(2, 0, 1, 3)` matches what downstream expects)

**Conclusion:** every kernel input I can verify in isolation is correct.
The kernel computes `cos > 0.99` against torch reference for the exact
same data shapes the e2e uses. Yet 32-layer e2e produces garbage tokens
(`OOOOak`) reproducibly. The bug is somewhere in the interaction
between the kernel and the rest of the model — possibly:

- Per-layer kernel invocation interacts with model state (residual
  stream, layer norms, MLPs) in a way one-shot unit tests can't catch.
- The output of fused SDPA has correct values but wrong tile/sharded
  layout that downstream ops (`nlp_concat_heads_decode`,
  `to_memory_config`) silently misinterpret.
- Some compounding precision / per-layer drift across 32 layers that
  exceeds what 0.99 cos suggests.

### 🎯 ROOT CAUSE FOUND (2026-04-28): missing positional mask

Logits comparison at step 41 (first generated token):

| | Baseline BFP8 | TQ Full Dequant |
|---|---|---|
| max | 33.5 | **9.5** |
| std | 2.39 | 1.77 |
| top-5 idx | 791="The", 60704, … | 20066="OO", 102470, … |

TQ logits are ~3× smaller magnitude with shifted mean. Post-mortem:

The fused-SDPA kernel iterates a fixed `k_chunk_size_tokens = 128`
positions per chunk. With `cur_pos=41` and `valid_k_chunks=1`, only
positions 0-41 in chunk 0 have real data; 42-127 are zero (paged_update_cache
never wrote them). For zero K/V positions:

- Q · 0 = 0 → score = 0
- exp((0 - real_max) * scale) ≈ exp(-0.66) ≈ 0.52 per zero position
- 86 zero positions × 0.52 = 44.7 added to softmax denominator
- 42 real positions contribute ≈ 21 to denominator
- Real positions get 21/65.7 ≈ 32 % of total weight (instead of ~100%)

V[zero] = 0 contributes nothing to output direction. So OUTPUT
DIRECTION is correct (matches torch ref → cos > 0.99 in unit tests),
but OUTPUT MAGNITUDE is ~32 % of correct.

LayerNorm partially compensates per-layer, but the consistent
~32 % bias compounds across 32 layers, eventually pushing the
residual stream into garbage. Std SDPA avoids this via the
`apply_mask` / `apply_causal_mask_lightweight` helpers which set
masked QK scores to -inf so exp() = 0. **Our fused TQ kernel has no
such mask.**

This explains every observation:
- All kernel inputs (cur_pos, Q, K_idx, K_norms, V_idx, V_norms) verified
  correct via DPRINT — kernel reads what it expects.
- Unit test cos > 0.99 — cosine is magnitude-insensitive, masks the bug.
- E2E garbage output — 32 layers of magnitude bias compound.
- BFP4 paged path works — std SDPA uses `cur_pos` to apply causal/padding
  mask, zeroing exp() contributions from positions > cur_pos.

### Fix plan

Add positional masking in the fused-SDPA kernel: after Q·K^T matmul
and before softmax, set scores at columns > cur_pos to -inf in the
last iterated chunk. Approaches in order of complexity:

1. **Reader-side mask CB** (chosen): allocate `cb_attn_mask` with one
   tile per Sk_chunk_t position. Reader computes mask from cur_pos
   and writes -inf (e.g. -1e10) for positions > cur_pos, 0 otherwise.
   Compute does `add_block_inplace(cb_qk_im, cb_attn_mask)` after
   matmul, before sub_exp.
2. **Compute-side mask via SFPU**: generate column-index tile in
   compute, apply `unary_ge_tile(col_idx, valid_cols)` × -1e10. More
   self-contained but needs more SFPU code.
3. **Iterate fewer K tiles per chunk**: clamp Sk_chunk_t dynamically
   to ceil((cur_pos+1)/32) tiles. Doesn't fix within-tile partial
   masking but reduces compute. Combine with (1) or (2) for the last
   tile.

Approach (1) is being implemented next.

### Bottleneck diagnosis trail (for posterity)

We initially suspected the centroid gather cascade (~50 SFPU ops/tile, 8
levels × 6 ops × 7 levels). Op-count math estimated ~1ms/layer for the
cascade, but trace measured 117 ms/layer — a 100× gap that pointed away
from the cascade. Profiling with `DeviceZoneScopedN` showed:

- `TQ_K_CHUNK_TOTAL` ≈ 442 us per chunk (cascade ~414 us of that)
- `TQ_FULL_DEQUANT_HEAD` ≈ 113 ms — implies ~256 chunks/head, not the 9 we
  expected. Confirmed via TQ_K_CHUNK_TOTAL count = 23,180 / 190 heads = 122
  observed (with overflow drops; actual is 256).
- 256 chunks × 442 us = 113 ms. Matches.

The 256 figure pointed to padded-cache iteration: max_num_blocks=1024 in
eval_e2e.py / Sk_chunk_t=4 = 256. Hypothesis test (hardcoding loop bound to
1) gave 6.0 ms/tok at 2L — confirmed.

The original Step 6 (centroid gather optimization, 6A/6B/6C alternatives)
becomes a smaller secondary opt now that the cap bounds the work to one
chunk for short seqs. Section 6 below still describes it for reference.

### Next steps (priority order)

1. **Diagnose output quality** — why does cap=1 give different output than
   cap=k_num_chunks when V data is zero for unfilled positions? Check:
   (a) does `paged_update_cache` for BFP4 actually write the right
   positions? (b) is the softmax accumulation state initialization wrong
   when only 1 chunk runs? (c) does `mm_init` need to be called more
   times? Add DPRINT to the kernel to dump cur_pos and per-chunk K data.
2. **Validate at long seq** — run with seq=2048 (16 chunks) and seq=8192
   (64 chunks). If those work, the cap=1 case is a special edge case.
3. **Step 5 accuracy benchmark** — once output is correct, re-run
   WikiText-2 perplexity / needle-in-haystack to confirm 3-bit quality.
4. **T3K validation** — should be a small lift now that perf is in range.

### Latest commits on `mtairum/kvcache_turboquant`

| Commit | What |
|--|--|
| `d7f7585a9da` | TQ SDPA: thread cur_pos into kernel; bound k_chunk loop (THIS IS THE FIX) |
| `7e6e166e715` | TQ SDPA: add DeviceZoneScopedN profiling zones to compute kernel |
| `b2edb19f03b` | PLAN: kernel-bottleneck diagnosis + Step 6 sketches (now superseded by this section) |
| `3da187312b0` | Step 1 (multi-chunk online softmax fix — cos 0.998+) |
| `c833e062f15` | Step 2 core (paged TQ cache + page_table-aware reader) |
| `39e590d6f03` | Step 2E (eval_e2e `--tq-full-dequant` flag + attention.py plumbing) |
| `a5160d2fd60` | Step 3A (mesh_mapper=ReplicateTensorToMesh for paged cache) |
| `ff548d2d19c` | Shared TQ cache across layers + `tq_layer_idx` attr |

### Useful repro commands

```bash
# Single device, specific layer count
timeout 240 python -u turbo_quant/eval_e2e.py --tq-full-dequant --num-layers 3 --max-new-tokens 5 --max-seq-len 128

# Same with --no-trace (note: no-trace times under-measure due to async dispatch)
timeout 240 python -u turbo_quant/eval_e2e.py --tq-full-dequant --num-layers 3 --max-new-tokens 5 --max-seq-len 128 --no-trace

# 4-device mesh
TT_NUM_DEVICES=4 timeout 300 python -u turbo_quant/eval_e2e.py --tq-full-dequant --num-layers 2 --max-new-tokens 5 --max-seq-len 128
```

Full details in Section 6 below.

---

## 1. Paper Reference & Summary

**TurboQuant**: Data-oblivious online vector quantization for KV cache compression.
- Paper: https://arxiv.org/html/2504.19874v1
- Target model: Meta-Llama-3.1-8B-Instruct on Tenstorrent Wormhole (N150)
- Model weights: `HF_HOME=/localdev/proj_sw/user_dev/hf_data`

### Result: same speed, half the KV cache, scales to 2,213 tok/s on T3K

3-bit TurboQuant with BFP4 paged cache vs baseline BFP8:

| | Baseline BFP8 | TurboQuant 3-bit BFP4 |
|--|--|--|
| **Single device (N150), batch=1** | 37.0 ms/tok | **37.2 ms/tok** (+0.2ms, +0.5%) |
| **T3K 8-device, batch=1** | 14.0 ms/tok | **14.2 ms/tok** (71 tok/s, 2.6× speedup) |
| **T3K 8-device, batch=32** | (KV-limited) | **2,213 tok/s** (14.5ms, 31× scaling) |
| **KV cache memory** | 1× (~1 byte/elem) | **0.5×** (~0.5 byte/elem) |
| **Qualitative quality** | — | Correct output at all seq lengths 128–131072 (31 prompts) |
| **Top-1 token accuracy** (T3K) | 96.7% | **72.5%** |
| **Top-5 token accuracy** (T3K) | 99.8% | **88.2%** |
| **Cosine vs CPU ref** | — | > 0.999 (synthetic SDPA test at all seqlens) |
| **MSE vs float32 CPU** | — | 0.034 (matches paper bound) |
| **Max context** | 128K | **128K** |

Verified 2026-04-14: BFP4 paged cache + standard SDPA decode, flat 37.1–37.2 ms/tok
from seq=128 to seq=131072. Pre-rescaled centroid×norm values stored as BFP4 in paged
`layer_past`, fed directly to `scaled_dot_product_attention_decode` which natively
accepts BFP4 inputs. No custom SDPA kernel needed for this path.

**Accuracy caveat (2026-04-20):** Rigorous token-accuracy benchmarking revealed
that BFP4 cache storage itself causes ~25% top-1 accuracy loss vs BFP8 baseline.
TurboQuant doesn't mitigate this — BFP4's shared-exponent compression dominates
the error. Qualitative tests (31 prompts, all factually correct) passed because
top-5 stays at 88-90%, enough for coherent output, but top-1 matching drops
substantially. See "Token Accuracy Benchmark" section below for full analysis.

### T3K Multi-Device Result (2026-04-17)

Running on T3K (8× Wormhole) with `TT_NUM_DEVICES=8` and `FABRIC_1D` config:

| | Single device (N150) | T3K (8 devices) | Speedup |
|--|---------------------|-----------------|---------|
| **Baseline BFP8** | 37.0 ms/tok | **14.0 ms/tok** | 2.6× |
| **TQ BFP4 paged** | 37.2 ms/tok | **14.2 ms/tok** | 2.6× |
| **TQ overhead** | +0.2ms (+0.5%) | +0.2ms (+1.4%) | constant |

**70.6 tok/s on T3K, flat across 128 → 131072 seqlens.** Change was 4 lines:
call `ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)` before opening the
mesh when `num_devices > 1`. TQ's `from_torch` calls replicate constants across
devices automatically. KV heads shard across devices: Llama-3.1-8B has 8 KV heads
→ 1 head/device on T3K.

### T3K Batch Throughput (2026-04-17)

Batch scaling on T3K with `--batch-size N`:

| Batch | Latency (ms/tok) | Throughput (tok/s) | Scaling vs batch=1 |
|-------|------------------|--------------------|--------------------|
| 1 | 14.2 | 70.6 | 1.00× |
| 4 | 14.1 | 283.4 | 4.01× |
| 8 | 14.1 | 565.9 | 8.01× |
| 16 | 14.2 | 1,128.2 | 15.98× |
| **32** | **14.5** | **2,213.2** | **31.35×** |

**Perfect linear scaling up to batch=32. 2,213 tok/s peak throughput.**
Latency barely grows (14.2 → 14.5ms). TQ's compressed BFP4 cache (0.5 bytes/elem,
2× smaller than baseline BFP8) enables these large batch sizes at long seqlens
without running out of DRAM — this is the key benefit of KV compression for serving.

### Token Accuracy Benchmark — BFP4 Storage is the Culprit (2026-04-20)

Rigorous token accuracy on the 1024-token reference corpus
(`models/tt_transformers/tests/reference_outputs/Llama-3.1-8B-Instruct.refpt`)
reveals BFP4 cache itself is lossy — **TurboQuant doesn't mitigate it**.

| Mode | Top-1 | Top-5 | KV memory | Latency |
|------|-------|-------|-----------|---------|
| **Baseline BFP8** | **96.7%** | 99.8% | 1× | 14.1 ms |
| TQ 3-bit + BFP8 cache | 89.1% | 100% | **1× (same!)** | 18.9 ms |
| Baseline BFP4 (no TQ) | 71.3% | 90.1% | 0.5× | 14.1 ms |
| TQ 3-bit + BFP4 cache | 72.5% | 88.2% | 0.5× | 18.9 ms |
| TQ 4-bit + BFP4 cache | 74.3% | 89.1% | 0.5× | 20.1 ms |

**TQ + BFP4 (72.5%) ≈ BFP4 alone (71.3%).** The BFP4 shared-exponent compression
(32-element blocks share 1 exponent) dominates the error — moving from 3-bit to
4-bit TQ barely helps (72.5% → 74.3%). TurboQuant's rotation + centroid approach
is being wasted when the storage format is BFP4.

**TQ + BFP8 is strictly worse than baseline BFP8:**
- Same memory (1 byte/elem, no savings)
- +4.8 ms latency overhead (+34%)
- -7.6% top-1 accuracy (96.7% → 89.1%)

No production use case where TQ + BFP8 wins.

**Qualitative vs Quantitative:** Our 31-prompt test showed factually correct
outputs for all TQ configs, matching the 82-88% top-5 numbers. Top-1 matches
baseline less often, but top-5 remains reasonable, which is enough for coherent
generation. But for rigorous accuracy (teacher-forced matching against a reference),
there's a significant gap.

**Implications:**
- Current "TQ BFP4 paged" path = half memory but ~25% top-1 accuracy loss
- TQ's design assumes high-precision storage (e.g. BF16); BFP4 breaks that assumption
- To recover accuracy, either use higher-precision cache (BFP8 or BF16) or accept
  the quality/memory tradeoff

### The Real Issue: Pre-Rescaled ≠ Paper's TQ Design

The low accuracy isn't inherent to TurboQuant — **our implementation took a shortcut
that breaks TQ's design guarantees**.

**Paper's design (Pure TQ / "Full Dequant"):**
- Store: 3-bit **integer index** (0..7) + BF16 **norm** (one float per vector)
- Memory: ~0.4 bytes/elem (3 bits/elem + 4 bytes per 128-elem vector)
- At read time: `centroid[idx] × norm` reconstructed with full precision
- **Accuracy preserved** — indices 0..7 are exact in any format; norm is BF16

**Our shortcut ("Pre-Rescaled BFP4"):**
- Compute `centroid[idx] × norm` at quantize time, store the product as BFP4
- Memory: ~0.5 bytes/elem (BFP4 storage)
- Storing continuous floats in BFP4 is lossy (shared-exponent per 32-elem block)
- **Accuracy lost** — BFP4's shared-exponent loses precision on pre-rescaled values

The paper's claims (MSE=0.034, cosine>0.999) apply to the **indices-plus-norm**
representation, NOT to pre-rescaled floats stored in BFP4. We conflated the two.

### Next Step: Revive Full Dequant as Main Path

Full Dequant (BFP4 indices + BF16 norms, fused SDPA with centroid reconstruction)
is the **correct TQ implementation**. We had this working previously but deprecated
it due to latency (15-35× slower than std SDPA). With the accuracy data we now have,
the tradeoff is worth revisiting:

- Full Dequant: ~0.4 bytes/elem, expected ~95%+ top-1 accuracy, slower SDPA
- Pre-Rescaled: ~0.5 bytes/elem, 72% top-1 accuracy, fast SDPA

For production, **correctness matters more than raw latency**. See Section 6 for
implementation plan.

### T3K Max Batch at Long Context (2026-04-20)

**The key serving benefit of KV compression:** TQ fits 2× the batch of baseline
at long sequence lengths where KV cache dominates DRAM usage.

| seqlen | TQ BFP4 max batch | Baseline BFP8 max batch | TQ throughput @ max | TQ advantage |
|--------|-------------------|-------------------------|--------------------:|--------------|
| 8K | 32+ (KV < 1GB) | 32+ | 2,222 tok/s | — |
| 32K | 32+ | 32+ | 2,217 tok/s | — |
| **64K** | **32** | 16 | **2,242 tok/s** | **2×** |
| **128K** | **16** | 8 | **1,136 tok/s** | **2×** |

Beyond max batch, baseline OOMs while TQ still runs. Baseline 64K×32 and 128K×16
both fail with DRAM OOM; TQ succeeds at the same config.

**Required fix:** Override `KV_CACHE` precision to BFP4 at **model init time**
(not just after init). Previously the model allocated BFP8 paged cache first,
limiting `max_num_blocks` to BFP8's memory footprint. See `make_optimizations`
in eval_e2e.py.

### T3K Multi-Device + Multi-Batch Quality (2026-04-17)

End-to-end quality verified on T3K with rotation-absorbed model and BF16 migration:

**10 diverse prompts (capitals, currencies, recipes, math, jokes, biology):**
All outputs factually correct, matches baseline. Spot-checks:
- Mix yellow + blue → green ✓
- 2+2 → 4 ✓
- Capital of USA → Washington D.C. ✓
- Capital of France → Paris ✓
- Currency of Brazil → BRL ✓

**Batch consistency check (batch=1 vs batch=4, same prompt):**
Bit-exact identical token IDs across all 20 generated tokens. No batching artifacts.

**Migration path:** Multi-device KV migration uses `ConcatMesh2dToTensor(dims=(0,1))`
to read sharded KV heads → full tensor, `ShardTensor2dMesh(dims=(None, 1))` to
write quantized values back to correct devices.

### E2E Overhead: TQ BFP4 vs Baseline BFP8 (2026-04-14)

Back-to-back comparison, same machine, same prompt, traced, 10 generated tokens.

| max_seq | Baseline BFP8 | TQ BFP4 Paged | Overhead |
|---------|--------------|---------------|----------|
| 128 | 36.9 ms/tok | 37.1 ms/tok | +0.2ms (+0.5%) |
| 1,024 | 36.9 ms/tok | 37.2 ms/tok | +0.3ms (+0.8%) |
| 4,096 | 37.0 ms/tok | 37.2 ms/tok | +0.2ms (+0.5%) |
| 16,384 | 37.0 ms/tok | 37.2 ms/tok | +0.2ms (+0.5%) |
| 65,536 | 37.0 ms/tok | 37.2 ms/tok | +0.2ms (+0.5%) |
| 131,072 | 37.0 ms/tok | 37.1 ms/tok | +0.1ms (+0.3%) |

Overhead is **constant O(1)** — only touches the 1 new token per step:
permute → centroid lookup (fused bucketize) → norm → pre-rescale (centroid×norm)
→ permute back → paged_update_cache scatter (BF16→BFP4 hardware conversion).
Rotation cost is zero (absorbed into W_v/W_o at model load time).

### Core Idea

TurboQuant compresses KV cache vectors during autoregressive LLM inference with
**no calibration data** and **near-optimal distortion**:

1. **Random rotation** (QR decomposition of Gaussian matrix) maps any vector to a
   known Beta distribution over coordinates
2. **Lloyd-Max quantization** per coordinate using precomputed codebooks
3. **Inner-product variant** adds 1-bit QJL on the residual for unbiased attention scores
4. Distortion guarantee: D_mse ≤ (√3π/2) · 4^(-b)

### Algorithm 1 — TurboQuant_mse

```
SETUP:
  Π ∈ ℝ^(d×d) via QR(randn(d,d))
  Precompute optimal centroids c₁,...,c_{2^b} via Lloyd-Max on Beta(d)

QUANTIZE(x):
  y = Π · x                          # rotate
  idx_j = argmin_k |y_j - c_k|       # nearest centroid per coordinate
  return idx, ||y||₂                  # b-bit indices + L2 norm

DEQUANTIZE(idx, norm):
  ỹ_j = c_{idx_j} · norm             # retrieve + rescale
  x̃ = Πᵀ · ỹ                         # rotate back
```

### Algorithm 2 — TurboQuant_prod (Inner-Product-Optimized)

```
QUANTIZE(x):
  idx, norm = mse_quantize(x)         # MSE at (b-1) bits
  r = x - mse_dequantize(idx, norm)   # residual
  qjl = sign(S · r)                   # 1-bit QJL
  return (idx, qjl, ||r||₂)

DEQUANTIZE:
  x̃ = mse_dequantize(idx) + √(π/2)/d · γ · Sᵀ · qjl
```

### Codebook Details

Coordinate distribution after rotation: f(x) = Γ(d/2)/(√π·Γ((d-1)/2)) · (1-x²)^((d-3)/2)

Precomputed MSE distortion per bit-width (d=128):

| Bits | Centroids | MSE (theoretical) | MSE (measured CPU) |
|------|-----------|--------------------|--------------------|
| 1 | 2 | 0.36 | 0.362 |
| 2 | 4 | 0.117 | 0.117 |
| 3 | 8 | 0.03 | 0.034 |
| 4 | 16 | 0.009 | 0.009 |

---

## 2. PyTorch CPU Implementation

### File Structure

```
turbo_quant/
├── __init__.py
├── rotation.py              # Random orthogonal matrix (QR decomposition)
├── codebook.py              # Lloyd-Max codebook on Beta distribution
├── quantizer.py             # TurboQuantMSE, TurboQuantProd, OutlierAwareTurboQuant
├── bitpack.py               # CPU bit-packing/unpacking for 1/2/3/4-bit indices
├── kv_cache.py              # HuggingFace-compatible KV cache wrapper
├── llama_integration.py     # HuggingFace Llama generation integration
├── ttnn_integration.py      # TTNN on-device implementation (Section 3)
├── eval_e2e.py              # End-to-end decode benchmark (teacher-forced + trace)
├── eval_e2e_prefill.py      # Real prefill + TurboQuant decode
├── eval_quality_comparison.py  # Side-by-side quality comparison
└── benchmarks/
    ├── test_correctness.py  # 34 CPU unit tests
    ├── test_ttnn.py         # 9 TTNN hardware tests
    ├── eval_perplexity.py   # Synthetic + real model perplexity
    ├── eval_latency.py      # Throughput and memory profiling
    ├── eval_needle.py       # Needle-in-a-haystack retrieval
    └── results_llama31_8b.json
```

### Components

| Module | What it does |
|--------|-------------|
| `rotation.py` | Generates Π ∈ ℝ^(d×d) via QR decomposition. Seed-based, orthogonality error < 1e-15. |
| `codebook.py` | Lloyd-Max algorithm for Beta distribution. Precomputed centroids for b=1,2,3,4. |
| `quantizer.py` | Three variants: MSE (Algorithm 1), Prod (Algorithm 2), OutlierAware (mixed bit-widths). |
| `bitpack.py` | Packs b-bit indices into uint8. 1-bit: 8×, 2-bit: 4×, 3-bit: 2.67×, 4-bit: 2×. |
| `kv_cache.py` | Drop-in replacement for HF `DynamicCache`. Supports all variants + beam search. |
| `llama_integration.py` | Monkey-patches `prepare_inputs_for_generation` for HF Llama models. |

### CPU Test Suite (34 tests, ~0.8s)

| Test Class | Count | Coverage |
|-----------|-------|---------|
| `TestRotationMatrix` | 3 | Orthogonality, determinism, seed variation |
| `TestCodebook` | 4 | Sorted centroids, symmetry, round-trip, index range |
| `TestTurboQuantMSE` | 4 | Shapes, monotonic MSE, zero input, single token |
| `TestTurboQuantProd` | 4 | Shapes, IP unbiasedness, IP error, min-bits validation |
| `TestOutlierAwareTurboQuant` | 7 | Effective bits, MSE ordering, calibration, configs |
| `TestBitPack` | 6 | Round-trip all bit-widths, compression, shapes, boundaries |
| `TestTurboQuantCache` | 6 | Prefill/decode, all variants, memory, bitpack |

Run: `PYTHONPATH=. python -m turbo_quant.benchmarks.test_correctness`

---

## 3. TTNN On-Device Implementation

### Fused Compute Kernels (C++)

Two custom TTNN device operations replace cascaded TTNN primitives with single-pass
SFPU kernels that operate entirely in DST registers (no DRAM intermediates):

**Fused Bucketize** (`turbo_quant_bucketize.cpp`):
Replaces 13 ops (7×ge + 6×add) with one kernel. Per tile: load input → loop over
boundaries using `copy_tile → unary_ge_tile → add_binary_tile`. ~31 SFPU ops/tile.

**Fused Gather Centroids** (`turbo_quant_gather_centroids.cpp`):
Replaces 21 ops (7×ge + 7×full_like + 7×where) with one kernel. Per tile:
conditional overwrite via `copy_tile → unary_ge_tile → fill_tile → sub/mul/add_binary_tile`.
~52 SFPU ops/tile.

Both use the multi-core program factory (`split_work_to_cores`) and reuse the
standard `reader/writer_unary_interleaved_start_id.cpp` dataflow kernels.

**Python API:**
```python
ttnn.experimental.turbo_quant_bucketize(input_tensor, boundaries)
ttnn.experimental.turbo_quant_gather_centroids(input_tensor, centroids)
```

Auto-detected at import time (`_FUSED_OPS_AVAILABLE`), graceful fallback to cascaded ops.

### Fused TQ SDPA Decode Kernel

Custom SDPA decode kernel that reads BFP4 quantization indices + BF16 norms from the
TQ cache and dequantizes on-the-fly during SDPA computation. Eliminates the full-cache
BF16 dequantize temporary that would otherwise be needed at long sequences.

**Pipeline (3 RISC-V cores per Tensix):**

```
Reader (RISC-V 0)           Compute (RISC-V 1-3)              Writer (RISC-V 4)
─────────────────           ──────────────────────             ─────────────────
Read Q (BF16) ──→ c_0       Pass 1: init_sfpu + typecast       Generate scale/
Read K idx (BFP4) → c_10      BFP4→BF16 into c_14              identity tiles
Read K norms (BF16)→ c_11   Pass 2: mm_init + centroid gather   Write output
Read V idx (BFP4) → c_12      + norm bcast_cols + K transpose    from c_16
Read V norms (BF16)→ c_13     → c_1 (K) and c_2 (V)
                             Then: sdpa_standard(c_0,c_1,c_2)
                               → c_16
```

**Key technical details:**
- `init_sfpu(src_cb, dst_cb)` required for BFP4→BF16 typecast (not just `copy_tile`)
- `mm_init()` must be called between SFPU (typecast) and FPU (bcast multiply) modes
- Centroid gather uses in-place `pack_tile<true>` to overwrite index tiles
- K tiles transposed during output via `pack_tile<true>(0, cb, col*Sk+row)`
- Reader reads K NOT transposed (row-major for norm alignment), compute transposes

**Python API:**
```python
ttnn.experimental.turbo_quant_sdpa_decode(
    q, k_indices, k_norms, v_indices, v_norms,
    page_table, cur_pos, centroids, scale
)
```

**Test results (cosine vs CPU reference):**
- 1 head: 0.9998, 8 heads: 0.9996, 32Q/8KV GQA: 0.9996

### C++ File Structure

```
ttnn/cpp/ttnn/operations/experimental/turbo_quant/
├── CMakeLists.txt
├── turbo_quant.hpp / .cpp            # Public API
├── turbo_quant_nanobind.hpp / .cpp   # Python bindings
├── device/
│   ├── turbo_quant_device_operation.hpp / .cpp
│   ├── turbo_quant_program_factory.cpp
│   └── kernels/compute/
│       ├── turbo_quant_bucketize.cpp
│       └── turbo_quant_gather_centroids.cpp
└── sdpa/
    ├── device/
    │   ├── sdpa_tq_device_operation.hpp / .cpp
    │   └── sdpa_tq_program_factory.cpp
    └── kernels/
        ├── compute/sdpa_tq_decode.cpp
        └── dataflow/
            ├── reader_tq_decode.cpp
            └── writer_tq_decode.cpp
```

### Key Optimisations

| Optimisation | What | Impact |
|-------------|------|--------|
| Fused kernels (B1+B2) | Single SFPU pass vs 13+21 TTNN ops | 71ms → 47ms |
| Rotated-space SDPA (B3) | Pre-rotate Q, post-rotate output, skip inverse rotation on full cache | +3ms at seq=512 |
| Cache centroid values | Gather at quantize time (1 token) not dequantize time (full cache) | 54ms → 49ms at seq=512 |
| Absorb Π into W_v/W_o | Bake rotation into projection weights (V has no RoPE) | Saves 64 matmuls/step |
| Pre-rescale centroids×norms | Store final values, dequantize = identity | O(1) dequantize |
| Paged BF16 cache | Use paged SDPA on pre-rescaled BF16 values | Flat latency, 128K context |
| BFP4 index cache | Integers 0-7 exact in BFP4 (~0.5 byte/elem) | 2× smaller than baseline |
| Fused TQ SDPA kernel | On-the-fly dequant inside SDPA (typecast+gather+norm) | No BF16 temp tensor |
| Pre-rescaled BFP4 SDPA | Typecast-only path: store centroid×norm as BFP4 | **= baseline latency, 4× less memory** |
| Multi-core fused SDPA | Distribute batch×heads across compute grid | ~8× speedup |
| rsqrt norm | Replace square+sum+sqrt+div with rsqrt | Fewer ops in quantize |
| Free layer_past | Deallocate BFP8 layer_past when TQ active | Doubles available DRAM |

### TTNN Hardware Test Suite (9 tests)

| Test | What it covers |
|------|---------------|
| `test_setup_tensors` | Rotation matrix upload, orthogonality on device |
| `test_quantize_shapes` | Index range [0, 2^b-1], output shapes |
| `test_dequantize_shapes` | Round-trip shape preservation |
| `test_roundtrip_quality` | MSE within BF16 bounds per bit-width |
| `test_cpu_reference_match` | Device vs CPU cosine > 0.70, index match > 90% |
| `test_monotonic_mse` | More bits → lower MSE (monotonicity) |
| `test_sdpa_decode_loop` | 4-step decode with real `scaled_dot_product_attention_decode` |
| `test_cache_update_and_dequantize` | Full cache scatter + dequantize pipeline |
| `test_latency` | Quantize + dequantize timing |

Run: `PYTHONPATH=. python turbo_quant/benchmarks/test_ttnn.py`

---

## 4. TurboQuant — Main Version

### Baseline BFP8 (no TurboQuant)

Standard Llama-3.1-8B inference path. BFP8 KV cache with paged SDPA.
No quantization overhead. Used as the performance/memory reference.

- **Flag:** `--no-turbo-quant`
- **KV memory:** ~1 byte/element (BFP8)
- **Latency:** 37ms/tok (flat across all seq lengths)

### TurboQuant BFP4 Paged Cache + Standard SDPA — THE MAIN VERSION

**This is the production TurboQuant variant.** Pre-rescales centroid×norm at quantize
time (O(1) per token), stores as BFP4 (~0.5 bytes/elem) in the model's paged KV
cache. Standard `scaled_dot_product_attention_decode` reads BFP4 natively — the
matmul unpacker handles BFP4→internal format conversion automatically. No custom
SDPA kernel needed. Rotation absorbed into W_v/W_o weights.

- **Flag:** `--bfp4-cache` in eval_e2e.py
- **KV memory:** ~0.5 bytes/element (BFP4) — **2× smaller than baseline**
- **Latency:** 37.1–37.2 ms/tok (flat, matches baseline)
- **Max context:** 128K verified (2026-04-14)
- **Cosine:** > 0.999 vs CPU reference at all seqlens
- **Status:** production-ready, verified at all seqlens 128–131072

### TurboQuant Pre-Rescaled BFP4 (fused SDPA) — DEVELOPMENT TRACK

Custom fused SDPA kernel that typecasts BFP4→BF16 on-the-fly with TQ-specific
dequant steps inside the kernel. Preserves fine-grained control over the dequant
pipeline. Currently limited to ~2K seq (L1 overflow at 4K+). Needs chunked online
softmax (Flash Attention style) to work at longer sequences — see Section 6.

- **Flag:** `pre_rescaled=True` in `turbo_quant_sdpa_decode`
- **KV memory:** ~0.5 bytes/element (BFP4) — **2× smaller than baseline**
- **Latency:** matches standard SDPA (0.03ms at seq=128, 0.17ms at seq=2048)
- **Cosine:** > 0.999 vs BFP4 roundtrip reference
- **Status:** multi-core, limited to ~2K seq (chunked dequant in progress)

### Deprecated Variants

The following variants are **deprecated** and should not be used for future
development or testing. They are retained here for historical reference only.

**TQ Performance (paged BF16 pre-rescaled)** — DEPRECATED
Stored pre-rescaled centroid×norm as BF16 in the model's paged KV cache.
Achieved latency parity at all seqlens up to 128K, but uses **2× baseline memory**
(BF16 = 2 bytes/elem vs BFP8 = 1 byte/elem). Superseded by the fused SDPA BFP4
variant which achieves 0.5× baseline memory instead.

**TQ Full Dequant (BFP4 indices + norms, fused SDPA)** — DEPRECATED
Stored BFP4 quantization indices + BF16 norms separately. Fused SDPA kernel did
centroid gather + norm multiply on-the-fly. Higher quality than pre-rescaled but
~15-35× slower due to per-tile centroid gather (~50 SFPU ops/tile). The quality
advantage does not justify the latency cost.

---

## 5. TTNN Experiments

All measurements on **Wormhole N150, Llama-3.1-8B-Instruct, batch=1, 3-bit**.

### Fused TQ SDPA Kernel Benchmark (multi-core) — MAIN VERSION

Synthetic data, 8Q/8KV heads, hd=128, 3-bit, Wormhole N150.
Verified 2026-04-14: passes 128–2048, fails 4096+ (L1 overflow at 34MB > 1.5MB).

| Seq Len | Pre-rescaled (ms) | Std SDPA (ms) | Cosine | BFP4 KV (MB) |
|---------|-------------------|---------------|--------|--------------|
| 128 | **0.03** | 0.03 | 0.9996 | 0.1 |
| 256 | **0.04** | 0.04 | 0.9996 | 0.2 |
| 512 | **0.05** | 0.06 | 0.9996 | 0.5 |
| 1024 | **0.09** | 0.11 | 0.9997 | 1.0 |
| 2048 | **0.17** | 0.20 | 0.9997 | 2.0 |
| 4096 | L1 limit | 0.41 | — | 4.0 |
| 8192 | L1 limit | 0.80 | — | 8.0 |
| 16384 | L1 limit | 1.57 | — | 16.0 |
| 32768 | L1 limit | 3.10 | — | 32.0 |
| 65536 | L1 limit | 6.21 | — | 64.0 |
| 131072 | L1 limit | 12.37 | — | 128.0 |

**Key findings:**
- **Pre-rescaled mode matches standard SDPA latency** — typecast-only path has negligible overhead
- Multi-core gives ~8× speedup over single-core (8 heads on 8 cores)
- L1 limit at 4K+: fused kernel pre-fills full BF16 cache in L1 CBs. Chunked dequant (Section 6) is the fix

### BFP4 K/V + Standard SDPA Decode — Synthetic Validation (2026-04-14)

The standard `scaled_dot_product_attention_decode` natively accepts BFP4 K/V inputs.
Pre-rescaled centroid×norm values stored as BFP4, fed directly to standard SDPA.
No custom kernel — standard SDPA's chunked online softmax handles all seqlens.

Synthetic data, 8Q/8KV heads, hd=128, 3-bit, Wormhole N150.

| Seq Len | Cosine vs CPU ref | Status | BFP4 KV (MB) |
|---------|-------------------|--------|--------------|
| 128 | 0.9996 | PASS | 0.1 |
| 256 | 0.9997 | PASS | 0.2 |
| 512 | 0.9997 | PASS | 0.5 |
| 1,024 | 0.9997 | PASS | 1.0 |
| 2,048 | 0.9997 | PASS | 2.0 |
| 4,096 | 0.9997 | PASS | 4.0 |
| 8,192 | 0.9997 | PASS | 8.0 |
| 16,384 | 0.9995 | PASS | 16.0 |
| 32,768 | 0.9996 | PASS | 32.0 |
| 65,536 | 0.9996 | PASS | 64.0 |
| 131,072 | 0.9995 | PASS | 128.0 |

**All 11 seqlens pass with cosine > 0.999.** No crashes, no L1 limit — standard SDPA
handles BFP4 at full 128K context. BFP4 shared exponent precision is sufficient for
pre-rescaled attention values.

### BFP4 Paged Cache — End-to-End Sweep (2026-04-14) — MAIN VERSION

Full Llama-3.1-8B-Instruct, teacher-forced decode, 3-bit TQ, traced, 10 generated
tokens. BFP4 paged `layer_past` + standard `scaled_dot_product_attention_decode`.
Correct output ("The capital of France is Paris.") at every seqlen.

| max_seq | TQ BFP4 Paged (ms/tok) | Warm avg (ms/tok) |
|---------|------------------------|-------------------|
| 128 | 37.2 | 37.2 |
| 256 | 37.2 | 37.2 |
| 512 | 37.2 | 37.2 |
| 1,024 | 37.1 | 37.1 |
| 2,048 | 37.2 | 37.2 |
| 4,096 | 37.2 | 37.2 |
| 8,192 | 37.1 | 37.1 |
| 16,384 | 37.1 | 37.1 |
| 32,768 | 37.1 | 37.1 |
| 65,536 | 37.2 | 37.2 |
| 131,072 | 37.1 | 37.1 |

**Flat 37.1–37.2 ms/tok across all seqlens.** Same speed as baseline BFP8, half the
KV cache memory (BFP4 = 0.5 bytes/elem). Works at full 128K context with no custom
SDPA kernel — standard SDPA's chunked online softmax handles BFP4 natively.

### KV Cache Memory (per batch, 32 layers × 2 K/V × 8 heads × 128 dim)

| max_seq | Baseline BFP8 (~1 B/elem) | TQ Pre-Rescaled BFP4 (~0.5 B/elem) | Savings |
|---------|--------------------------|-------------------------------------|---------|
| 2,048 | 143 MB | **72 MB** | 2× |
| 4,096 | 285 MB | **145 MB** | 2× |
| 8,192 | 570 MB | **289 MB** | 2× |
| 16,384 | 1.1 GB | **579 MB** | 2× |
| 32,768 | 2.3 GB | **1.2 GB** | 2× |
| 65,536 | 4.6 GB | **2.3 GB** | 2× |
| 131,072 | 9.1 GB | **4.6 GB** | 2× |

### Quality (real prefill + decode, greedy sampling)

| Prompt | Bits | Output | Correct? |
|--------|------|--------|----------|
| "What is the capital of France?" | 2 | "The capital of France is Paris." | Yes |
| "What is the capital of France?" | 3 | "The capital of France is Paris." | Yes |
| "What is the capital of France?" | 4 | "The capital of France is Paris." | Yes |
| Quantum computing (3 sentences) | 2 | Correct: superposition, entanglement, parallel processing | Yes |
| Quantum computing (3 sentences) | 3 | Correct: same topics, slightly different wording | Yes |
| Number sequence (continue 90..99) | 3 | Correctly outputs "100" | Yes |
| Megaliths (188-tok, avoid keywords, `******` separator) | 3 | Follows all constraints | Yes |
| Grafton VT (3 paragraphs, start with "send") | 3 | Follows all formatting | Yes |

### WikiText-2 Perplexity + KV Cache Distortion (2026-04-14)

Llama-3.1-8B-Instruct, BF16 on CPU, 4550 tokens, sliding-window PPL.
KV distortion measured on 5 captured windows × 32 layers of real KV tensors.

**Baseline perplexity: 9.91**

| Variant | Key MSE | Key Cosine | Value MSE | Value Cosine |
|---------|---------|------------|-----------|--------------|
| MSE 2-bit | 0.4725 | 0.9400 | 0.0123 | 0.9401 |
| **MSE 3-bit** | **0.1384** | **0.9828** | **0.0036** | **0.9828** |
| MSE 4-bit | 0.0380 | 0.9953 | 0.0010 | 0.9953 |
| Outlier 2.25-bit | 0.3893 | 0.9508 | 0.0101 | 0.9510 |

3-bit (production config): cosine 0.983 on real model KV tensors, matches paper bound.
Monotonic quality: 2-bit → 3-bit → 4-bit. Outlier 2.25-bit outperforms plain 2-bit.

### Needle-in-a-Haystack Retrieval (2026-04-14)

Synthetic test: planted needle key with known high affinity, measured retrieval after
TQ quantize/dequantize. 5 needle positions × 6 haystack lengths × 6 variants = 180 tests.

| Variant | Retrieval Accuracy | Haystack lengths |
|---------|-------------------|------------------|
| FP32 baseline | 30/30 (100%) | 64, 256, 1K, 4K, 16K, 64K |
| MSE 2-bit | 30/30 (100%) | " |
| MSE 3-bit | 30/30 (100%) | " |
| MSE 4-bit | 30/30 (100%) | " |
| Prod 3-bit | 30/30 (100%) | " |
| Outlier 2.25-bit | 30/30 (100%) | " |

**100% retrieval across all variants**, including 2-bit, at all context lengths up to 64K.

### Quality Comparison: 31 Diverse Prompts (2026-04-14)

Side-by-side baseline BFP8 vs TQ 3-bit on 31 prompts (capitals, currencies, recipes,
jokes, math, biology, sports, travel). Prefill + 100 tokens decode per prompt.

- **All 31 prompts produce factually correct, coherent output** — no hallucinations,
  no repetitive text, no quality degradation from TQ quantization
- **Average word overlap: 81%** (range 57–96%)
- No exact matches expected: rotation absorption changes model wording slightly
- Spot-checks: capitals (Washington D.C., Ottawa, London, Berlin, Paris, Tokyo,
  Lisbon, Beijing), currencies (all correct), math (2+2=4), color mixing (green)

### Optimisation History

```
183ms/tok   initial TurboQuant (A0)
 71ms/tok   TTNN trace (A2t)
 47ms/tok   fused kernels (B1+B2)
 46ms/tok   cache centroids
 45.6ms     absorb Π into W_v/W_o
 44.1ms     pre-rescale centroids×norms
 43.5ms     rsqrt norm + remove UINT32 typecast
 37.2ms     paged BF16 with paged SDPA (= baseline, 2 bytes/elem)
 37.2ms     BFP4 index cache (= baseline, FLAT 128→131072, 0.5 bytes/elem)
  0.17ms    fused BFP4 SDPA pre-rescaled @ seq=2048 (= baseline, 0.5 bytes/elem)
 37.1ms     BFP4 paged cache + standard SDPA (= baseline, FLAT 128→131072, 0.5 bytes/elem)
```

---

## 6. Constraints & Next Steps

### Status Summary

| | Latency | KV Memory | Top-1 Acc | Max Context | Status |
|--|---------|-----------|-----------|-------------|--------|
| Baseline BFP8 | 1× | 1× | **96.7%** | 128K | Reference |
| TQ BFP4 Paged (pre-rescaled) | 1× | 0.5× | 72% | 128K | Works but lossy (2026-04-20) |
| **TQ Full Dequant (indices + norms)** | **TBD** | **~0.4×** | **TBD (expect ~95%+)** | TBD | **NEW MAIN TARGET** (see below) |

### Prefill → BFP4 decode: paged prefill path (follow-up)

`eval_e2e_prefill.py --bfp4-cache` works with non-paged model: prefill (BFP8) →
migrate (BFP4) → decode (BFP4, non-paged SDPA). Output is correct ("Paris.") with
no repetitive output issue. Decode latency is ~43ms/tok (non-paged SDPA) vs 37ms
(paged SDPA in eval_e2e.py).

To get 37ms decode after prefill, the model needs paged attention for decode. Paged
prefill currently fails with block_size mismatch (`Input tensor height (128) must be
<= cache tensor height (32)`). This is a model-level issue with how `prepare_inputs_prefill`
handles paged KV caches, not a TurboQuant issue. Follow-up: investigate paged prefill
setup or two-phase model init (non-paged prefill → paged decode).

### NEW MAIN TARGET: Full Dequant (BFP4 indices + BF16 norms) — REVIVED (2026-04-20)

After the accuracy benchmark revealed the pre-rescaled path loses ~25% top-1
accuracy due to BFP4 shared-exponent compression, we're reviving the Full Dequant
path as the correct implementation of TurboQuant. This is the paper's original
design: store quantization indices + per-vector norms separately, reconstruct at
read time. Previously deprecated for latency; now justified by accuracy data.

**What exists already:**
- Custom fused SDPA kernel at `ttnn/cpp/ttnn/operations/experimental/turbo_quant/sdpa/`
  - Device op + program factory + reader + compute + writer (~925 lines C++)
  - Reads BFP4 indices + BF16 norms, does centroid gather + norm multiply in compute
  - Python binding: `ttnn.experimental.turbo_quant_sdpa_decode`
- TTNNTurboQuantCache with `memory_efficient=True` stores BFP4 indices + BF16 norms
- Synthetic tests pass with cosine > 0.999

**Known limitations:**
- Works up to 2K seq; pre-fills full BF16 cache in L1 CBs → OOMs at 4K+
- Uses interleaved tensors (not paged) — separate `k_indices_dev` / `k_norms_dev`
- 15-35× slower than std SDPA (at seqs that fit) due to centroid gather overhead
- Untested on T3K / multi-device

**Implementation Plan:**

**1. Chunked online softmax in fused kernel** (critical, unblocks >2K context) — **DONE (2026-04-24)**

- Modified `sdpa_tq_decode.cpp` to stream K/V chunks instead of pre-filling
- CBs shrunk from `k_num_chunks` to 2 (double-buffer) → no OOM at 4K+
- New per-chunk dequant helpers (`dequant_k_chunk`, `dequant_v_chunk`) that
  match tile layout (transposed K, natural V) for matmul_blocks compatibility
- Interleaved: dequant one K/V chunk → matmul Q×K^T → softmax → V matmul → accumulate

**Multi-chunk bug fix (commit `3da187312b0`, 2026-04-24):**
Root cause was a PACK→UNPACK visibility gap on the ping-pong softmax-state CBs
(cb_max_A/B, cb_sum_A/B, cb_out_im_A/B). PACK writes these via pack_tile+push_back
in iteration N, but when UNPACK reads them as prev_max/sum/out in iteration N+1,
the L1 data hasn't fully settled despite cb_push_back's TTI_STALLWAIT.

Discovered via bisection: adding `DPRINT_UNPACK(DPRINT << TSLICE(...))` at
end-of-iteration fixed it up to 8 chunks. Translated to non-DPRINT form as
`sync_unpack_cb_read()` — a noinline function doing 64 volatile uint32 L1 reads
on UNPACK thread, wrapped in `UNPACK(...)` macro. Reading only 8 words fails;
64 is the empirical minimum.

**Validation (all PASS, cos ≥ 0.9983):**
| seqlen | k_chunks | cos |
|--------|---------:|----:|
| 128    |     1    | 0.9996 |
| 2048   |    16    | 0.9997 |
| 8192   |    64    | 0.9996 |
| 32768  |   256    | 0.9995 |
| 131072 |  1024    | 0.9983 |

**2. Paged cache support for indices + norms** — **DONE (2026-04-24)**

Current state (from post-Step-1 inspection):
- `TTNNTurboQuantCache` allocates `k/v_indices_dev` as `[B, NKH, max_seq_padded, DH]`
  BFP4, and `k/v_norms_dev` as `[B, NKH, max_seq_padded, 1]` BF16 — **contiguous, NOT paged**
- Model's standard `layer_past` is paged `[max_num_blocks, NKH, block_size=32, DH]`
  with a separate page_table
- TQ cache + standard paged cache both allocated — not unified
- `fused_sdpa_decode()` passes a dummy page_table (unused by current reader)
- `sdpa_tq_device_operation` has `page_table` field in tensor_args but reader
  kernel ignores it

Sub-phases:

**2A.** Refactor `TTNNTurboQuantCache` to paged layout:
   - Indices: `[max_num_blocks, NKH, block_size, DH]` BFP4
   - Norms: `[max_num_blocks, NKH, block_size, 1]` BF16
   - Accept `paged_attention_config` + `page_table` in `__init__` / `update_cache`
   - Reuse existing `ttnn.experimental.paged_update_cache` with `page_table=` arg
     (already supported — just need paged tensor shapes)

**2B.** Thread page_table through `fused_sdpa_decode()`:
   - Accept real page_table argument (no more dummy)
   - Pass to TQ SDPA op via tensor_args

**2C.** Update reader kernel (`reader_tq_decode.cpp`) for paged reads:
   - Mirror `reader_decode_all.cpp` pattern: read page_table,
     use `virtual_seq_tile_id_to_physical_tile_id` to translate chunk start → physical tile
   - Keep index-vs-value CB routing logic for pre_rescaled mode

**2D.** Update program factory (`sdpa_tq_program_factory.cpp`):
   - Pass page_table buffer address to reader runtime args
   - Add compile args (is_page_table_sharded, block_size_t)
   - Increase kernel CT args count

**2E.** Update `eval_e2e.py` to use unified paging — **DONE (2026-04-24)**
   - Added `--tq-full-dequant` flag
   - Allocates TQ cache with `paged_config` matching model's paged_attention_config
   - Frees model's standard `layer_past` (BFP8 paged cache) since fused path doesn't use it
   - `attention.py` forwards `page_table` through `turbo_quant_cache.update_cache()`
     and `.fused_sdpa_decode()` (paged mode); also permutes Q from [1,B,NQH,DH] to
     [B,NQH,1,DH] as the TQ op expects, and permutes output back.
   - End-to-end validated with 2 layers: 6.9 ms/tok, 144.5 tok/s single-seq
   - Full 32-layer test deferred (only slow due to 32× cold-cache compile; logic verified
     at 2 layers)

**Step 2 COMPLETE (2026-04-24, commits `c833e062f15` + `39e590d6f03`).**
Paged TQ cache + page_table-aware reader + e2e integration all working on
single-device N150. Next major piece is Step 3 (multi-device / T3K).

Quick-reference status after Step 1+2:

| Layer | Status | Commit | Validation |
|-------|--------|--------|------------|
| Multi-chunk online softmax | ✅ DONE | `3da187312b0` | cos 0.9983+ at seq 128–131072 |
| TTNNTurboQuantCache paged layout | ✅ DONE | `c833e062f15` | standalone test passes |
| Reader kernel paged reads | ✅ DONE | `c833e062f15` | identity + shuffled page_tables |
| fused_sdpa_decode page_table plumbing | ✅ DONE | `c833e062f15` | standalone test cos 0.9997 |
| eval_e2e.py --tq-full-dequant | ✅ DONE | `39e590d6f03` | 2-layer e2e: 6.9 ms/tok |
| T3K multi-device | ⏳ PENDING | — | Step 3 |
| Accuracy/latency benchmarks | ⏳ PENDING | — | Step 5 |

**Performance benchmarks (2026-04-24, N150 single device, --no-trace):**

| Layers | Warm avg ms/tok | Per-layer cost (added) | Notes |
|-:|-:|-:|-|
| 2 | 7 | 3.5 ms | Fast path (small-model-compatible?) |
| 3 | 121 | +114 ms | Jump: per-layer cost ~117 ms after layer 2 |
| 4 | 238 | +117 ms | Consistent with 3L→4L |
| 8 | 707 | +117 ms/layer | Linear from 3L onward |
| 32 | ~3500 (extrapolated) | +117 ms/layer | ~100× slower than baseline |

**Comparison points:**
- Baseline BFP8 paged (32L): ~37 ms/tok (~27 tok/s)
- TQ Pre-rescaled (32L, prod): 37.2 ms/tok (flat to 128K)
- TQ Full Dequant (32L, projected): ~3500 ms/tok ≈ **0.3 tok/s**

**🚨 Scaling pattern** — first 2 layers essentially free (7 ms total), then a step
up of ~117 ms per additional layer. The 2→3 layer jump (7→121 ms) looks like a
code-path switch, not simple arithmetic. 32-layer runs for >2 minutes per token
(earlier "hang" was just slow decode finishing within timeout for small seqlen).

Shared-cache refactor (commit `ff548d2d19c`): allocated one TTNNTurboQuantCache
with `num_layers=N` instead of N separate instances with `num_layers=1`. Cleaner
architecturally but did NOT change perf (same 238 ms/tok at 4L). So the
non-linear jump isn't per-layer cache allocation — it's something structural in
the fused-SDPA kernel invocation path at ≥3 layers.

**Leading hypotheses** (need profiling to confirm):
- Program cache may hash on buffer address or tensor identity, causing miss
  even with identical shapes/layouts across layers
- `sync_unpack_cb_read` helper (Step 1 fix) has a fixed 64-word volatile read
  which may become a bottleneck under repeated invocation
- Python→C++ dispatch overhead in `ttnn.permute` and `ttnn.experimental.paged_update_cache`
  compounds with layer count (trace-mode could fix this but it hangs for other reasons)

**Multi-device (2026-04-24, N150×4 mesh):** Added `mesh_mapper=ReplicateTensorToMesh(...)`
in `TTNNTurboQuantCache.__init__` so the paged index/norm caches replicate properly
on MeshDevice. Verified model load → cache alloc → compile (22.1 s compile time)
work on 4-device mesh. Decode hangs at loop start for 2-layer test — same scaling
issue as single-device layers×devices total-invocation count (2 layers × 4 devices
= 8 cumulative invocations per step). Need to fix the core scaling issue before
T3K validation is meaningful.

**Followups before accuracy runs can be meaningful:**
1. ~~Profile the 3L run with Tracy to identify where the ~117 ms/layer cost is spent~~
   Trace-mode probe (2026-04-27) localizes the cost to the kernel itself — see
   resume-block at top of plan for diagnosis.
2. ~~Test with `--tq-full-dequant` + trace mode after resolving the trace-mode hang~~ DONE
3. ~~Check program cache hit rate~~ DONE — cache holds at 55 entries, no thrash
4. ~~If dispatch-bound, reduce Python-side permute overhead~~ Not the bottleneck

---

### Step 6: Centroid gather optimization — REQUIRED (was Optional)

The trace-mode results elevate Step 6 from "optional" to "**critical path**".
Without it the Full Dequant path is ~100× slower than baseline and not
production-viable — Steps 3 (T3K) and 5 (accuracy/latency) become moot until
Step 6 lands.

#### Where the cost is

`sdpa_tq_decode.cpp::dequantize_one_tile()` (lines 55–111) contains the hot
loop. For `NumLevels=8` (3-bit centroids):

```
Phase 1 (centroid gather, per tile):
  copy_tile                      // 1
  fill_tile                      // 1
  for lev in 1..7:               // 7 iterations × 6 ops = 42 ops
      copy_tile + unary_ge_tile + fill_tile +
      sub_binary_tile + mul_binary_tile + add_binary_tile
Phase 2 (norm multiply, per tile):
  mul_tiles_bcast_cols           // 1
                          Total: ~46 SFPU op-issues per tile
```

This runs on **every K and V tile, every chunk, every layer, every token**:

```
chunk_tiles  = Sk_chunk_t × DHt   = 8 × 4   = 32  (typ. for head_dim=128)
per-chunk    = 32 K + 32 V        = 64 tiles
seq=128      = 4 chunks           = 256 tiles/layer
× 32 layers  = 8192 tiles/tok
× 46 ops/tile= ~377K SFPU op-issues per token
```

At seq=8K it's 16× more (~6M SFPU ops/tok). The SFPU pipeline plus its
per-issue dispatch overhead is the dominant cost in the trace-measured
117 ms/layer.

#### Approach 6A — LUT matmul gather (recommended first, ~3 days)

**Idea.** Replace the 8-level conditional cascade with one matmul. The gather
is mathematically `centroid_value = onehot(idx) · centroids_vector`. If we
expand a tile of indices into a `[32, NumLevels]` one-hot and matmul against a
`[NumLevels, 32]` constant centroid LUT broadcast across head-dim, we get the
gather as **one FPU matmul tile** instead of ~46 SFPU ops.

**Sketch:**
```cpp
// New helper: expand BFP4 idx tile (32×32 indices in 0..7) → one-hot [32, 8]
// Implementation option: SFPU `eq_binary_tile` against 8 level constants,
// packing each result into one of 8 output columns. Costs 8 ops, but only
// once per tile — vs 46 in the cascade.
expand_indices_to_onehot(cb_idx, cb_onehot);    // [32 rows × 8 cols] BF16

// Pre-load constant LUT tile: [8 rows × 32 cols] where row k = centroids[k]
// replicated across columns. This lives in DRAM as a baked compile-time
// tensor (one [8,32] BF16 tile per layer; identical across layers if shared
// centroids).
matmul_tiles(cb_onehot, cb_centroid_lut, dst);   // 1 FPU matmul → [32×32] gathered

// Phase 2 norm multiply unchanged
mul_tiles_bcast_cols(cb_gathered, cb_norms, ...);
```

**Cost estimate.** Matmul of two `[32×32]` tiles on Wormhole FPU is ~1 cycle
under saturation (vs ~46 SFPU op-issues). Even with the 8-op one-hot expansion
overhead, **expected ~5×** kernel-arithmetic speedup. If we can fold the
one-hot expansion into the unpacker (BFP4 → one-hot via a small lookup
table at unpack time), the win goes higher.

**Risks.**
- BFP4 unpack semantics: BFP4 values come in pre-scaled as floats; the index
  0..7 lives in the mantissa bits. Need to verify `copy_tile` from a BFP4 CB
  yields integer-valued floats we can `eq_binary_tile` against.
- Constant LUT must be re-uploaded per layer if centroids differ per layer.
  If centroids are shared (one global codebook), bake it as a kernel CT-arg
  array — same as today's `centroids[16]`.
- One-hot expansion needs validation: SFPU `eq_binary_tile` produces 0/1
  outputs that must be packed into the right column of the output tile.

**Validation.** Reuse `bench_fused_sdpa.py`. Pass criterion: cos ≥ 0.998 at
seq 128–131072 (matches Step 1 baseline) **and** ≥3× speedup vs current
kernel at trace-measured 3L decode.

**Files.**
- `sdpa_tq_decode.cpp::dequantize_one_tile` — replace cascade with onehot+matmul
- `sdpa_tq_decode.cpp::dequant_k_chunk`, `dequant_v_chunk` — same, in their
  inlined cascades (lines ~175–215 for K, ~253–290 for V)
- `sdpa_tq_program_factory.cpp` — add LUT CB allocation + initialization
- `reader_tq_decode.cpp` — read centroid LUT once (or via CT-arg if static)

#### Approach 6B — Post-softmax centroid expansion (largest win, ~1 week)

**Idea.** Don't expand indices to BF16 K/V at all. Keep K and V as BFP4
indices through the Q×Kᵀ and softmax×V matmuls. The math is:

```
Q × Kᵀ = Q × (centroids[K_idx] ⊙ K_norm)ᵀ
       = (Q ⊙ K_norm)  ×  centroids[K_idx]ᵀ      (norm is per-row; broadcasts)
       = (Q ⊙ K_norm)  ×  C[K_idx]ᵀ
```

We can rewrite Q×Kᵀ as `(Q · centroids_packed_as_LUT)[K_idx]` — the matmul of
Q against the **8 centroids** (a tiny `[Q_rows × 8]` matrix), then a per-tile
scatter using K_idx. This makes Q×Kᵀ ~32× cheaper because the inner dim
collapses from `head_dim=128` to `NumLevels=8`.

Same trick for softmax × V. The savings compound across both matmuls and the
gather.

**Sketch:**
```cpp
// Phase 1: Q · centroidsᵀ → small [Q_rows × NumLevels] tile (constant for
// all K positions in this Q step). Done once per chunk.
matmul_tiles(cb_q, cb_centroids_lut_T, cb_q_dot_centroids);  // [B*NQH × 8]

// Phase 2: scatter-gather: for each K row, look up indices and select from
// the precomputed Q·centroids vector, multiply by K_norm.
// This is essentially a "matmul against one-hot" step but the inner data is
// already reduced.
scatter_gather_qk(cb_q_dot_centroids, cb_k_idx, cb_k_norms, cb_qk_scores);

// Same trick for softmax × V on the output side.
```

**Cost estimate.** Q×Kᵀ inner-dim drops from 128 to 8 → **~16× cheaper matmul**
plus the gather is now a scatter (no SFPU cascade at all). Combined gain:
expected **8–15× speedup**. This brings projected 32-layer decode to ~250ms,
within striking distance of baseline (37ms) given the BFP4 memory savings.

**Risks.**
- Significant kernel rewrite — affects all four phases of fused SDPA decode
  (gather, Q×Kᵀ, softmax, softmax×V) and the matmul block layouts.
- The "scatter via index" step has no direct primitive on Tensix — needs to
  be expressed as masked sums or an SFPU lookup. Empirical: probably needs
  a custom data-flow pattern in unpacker.
- Numerical: applying norm post-matmul changes accumulation order. Need to
  verify cos ≥ 0.998 still holds.

**Validation.** Same as 6A plus a numerical equivalence test against the
current kernel at low seqlens to confirm reordering doesn't drift.

**When to do.** Only if 6A doesn't hit the latency target (i.e. if 6A gives
<3× speedup). 6B is harder but uncaps the win.

#### Approach 6C — Vectorize the cascade (smallest change, smallest win, ~1 day)

**Idea.** Keep the cascade structure but parallelize the 8 levels. Tensix
SFPU has 32-wide SIMD. Today's cascade serializes on the cross-tile
dependencies (each `add_binary_tile` reads the previous level's accumulator).
We can compute all 8 level contributions independently and reduce them in
log₂(8)=3 levels of pairwise add.

**Sketch:**
```cpp
// Level contributions in parallel (8 independent fill+sub+ge+mul):
fill_tile(R0, centroids[0]);
for k in 1..7:
    fill_tile(Rk, centroids[k]);
    sub_binary_tile(Rk, R0, Rk);          // delta_k = c_k - c_0
    unary_ge_tile(idx, level_bits[k]) → mask_k;
    mul_binary_tile(Rk, mask_k, Rk);

// Log-tree reduction:
add_binary_tile(R1, R2, R1);  add_binary_tile(R3, R4, R3);
add_binary_tile(R5, R6, R5);  add_binary_tile(R7, R0, R7);  // R0 absorbed here
add_binary_tile(R1, R3, R1);  add_binary_tile(R5, R7, R5);
add_binary_tile(R1, R5, R1);  // result in R1
```

**Cost estimate.** Same op count, but with 8 DST registers in flight and the
log-tree reduction the critical path is ~8 op-issues + 3 add-issues = ~11
ops vs 46 today. **Expected ~3× speedup** of the gather phase. Norm multiply
is unchanged.

**Risks.**
- DST register pressure: need 8+ DST tiles concurrently. Wormhole has 8 DST
  tiles per math thread; 8 levels × 1 tile each saturates the file. May
  need to spill to L1.
- Less aggressive than 6A/6B; may not be enough.

**When to do.** If 6A turns out infeasible (BFP4 unpack issue blocks
one-hot expansion), 6C is the conservative fallback.

#### Recommended path

1. Implement **6A (LUT matmul gather)**, validate cos + perf at 3L. Target:
   ≥3× speedup → 32L extrapolation under ~1 second/tok.
2. If 6A meets the bar, run Step 5 (accuracy benchmark + latency on T3K).
3. If 6A doesn't suffice, layer 6B on top; 6A's onehot+LUT primitives are
   reusable.
4. Ship 6C only as a fallback if 6A is blocked.

---

### Steps 3–5 (deferred until Step 6 lands)

**3. Multi-device / T3K support** — DEFERRED behind Step 6
- Replicate centroids across devices, shard indices/norms by heads
- `fused_sdpa_decode` to use mesh_composer/mesh_mapper where needed
- Verify on 8-device T3K with FABRIC_1D (same setup as pre_rescaled path)
- Code already mesh-compatible (commit `a5160d2fd60`); blocked on perf
- Effort: 1 day after Step 6

**4. Full validation of single-device path** — PARTIAL (basic e2e validated)
- ✅ `--tq-full-dequant` flag present in eval_e2e.py
- ✅ 2-layer end-to-end decode works
- ⏳ Full 32-layer validation (deferred until per-layer cost is reasonable)
- ⏳ Port flag into `eval_e2e_prefill.py` for real prefill → decode
- ⏳ Quality comparison vs baseline/pre-rescaled at full 32L
- Effort: 1-2 days after Step 6

**5. Accuracy + latency validation** — DEFERRED behind Step 6
- Run token accuracy benchmark → expect >95% top-1 (vs pre-rescaled's 72%)
- Measure decode latency → target <30ms on T3K (2× baseline acceptable given accuracy)
- Run seqlen + batch sweeps to confirm long context + multi-batch works
- Effort: 1-2 days after Step 6

**Total effort estimate for remaining work: 1-2 weeks (Step 6 is ~3 days
to 1 week depending on which approaches land; Steps 3-5 are ~3-5 days
afterwards).**

**Alternative approach (simpler but higher risk):**
Modify the standard SDPA decode reader to support TQ dequant on-read. Reuse std
SDPA's proven chunked online softmax. Only the reader needs TQ logic. Risk:
touching shared infrastructure used by other models.

### ~~Prefill → TQ decode migration (quality issue)~~ RESOLVED

Previously, migrating prefill BFP8 KV into TQ format produced repetitive output.
**Fixed as of 2026-04-14:** `eval_e2e_prefill.py --bfp4-cache` produces correct,
coherent output ("The capital of France is Paris.") with no repetition. The fix
was storing centroid×norm (not just centroids) and using BFP4/BF16 cache.

### ~~Quality benchmarks~~ DONE (2026-04-14)

All completed:
- **WikiText-2 perplexity:** baseline 9.91, 3-bit KV cosine 0.983, MSE 0.138
- **Needle-in-a-haystack:** 100% retrieval across all variants (2/3/4-bit) up to 64K
- **31-prompt quality comparison:** all factually correct, 81% avg word overlap vs baseline

### Remaining Next Steps

**~~End-to-end demo with real prefill + paged decode~~ DONE (2026-04-20):**

`eval_e2e_prefill.py --bfp4-cache` now runs real prefill + paged decode on device,
no teacher forcing. Correct output across prompts.

Fix was passing `page_table=` to `prepare_inputs_prefill` and `kv_cache=` to
`ttnn_prefill_forward` (previously omitted). Migration function now handles paged
layer_past layout `[max_num_blocks, n_kv_heads, block_size, head_dim]`.

| Config | Prefill+decode (ms/tok) | eval_e2e.py teacher-forced | Gap |
|--------|-------------------------|---------------------------|-----|
| Single device | 42.8 | 37.1 | +5.7ms |
| T3K 8-device | 18.7 | 14.2 | +4.5ms |

**Residual ~5ms gap vs teacher-forced path** — possibly program cache / memory
fragmentation after prefill. Non-blocking but worth investigating. Not TQ-specific
(would likely affect any paged-prefill → paged-decode flow).

**Max batch at 128K context:**
Sweep to find batch limit at long seqlen. TQ's 2× KV compression should enable
significantly larger batches than baseline BFP8 at extended context.

**Galaxy (TG, 32 devices):**
Untested. Should work with same `FABRIC_1D` config. Would give ~4× T3K throughput.

**Formal accuracy benchmarks:**
MMLU, HellaSwag, or similar — for a formal "accuracy retained" number.

**~~Multi-batch~~ DONE (2026-04-17):**
- T3K batch sweep: perfect linear scaling 1→32, 2,213 tok/s peak at batch=32

**~~Multi-device~~ DONE (2026-04-17):**
- T3K (8× Wormhole) verified: 14.2 ms/tok, 2.6× speedup vs single device

**~~Fused SDPA kernel~~ DROPPED (2026-04-17):**
- Redundant with BFP4 paged path. C++ sources retained for reference only.

**True 3-bit packing:**
Pack indices to 3 bits/element (0.375 bytes) instead of BFP4's ~0.5 bytes.
Requires custom pack/unpack kernels + ROW_MAJOR scatter.
