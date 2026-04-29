# Sliding-Window Hybrid (TQ + BFP8 ring) — Design Doc

**Goal:** recover the −10 pp top-1 gap of Track A (TQ full-dequant) while
keeping the 0.75× KV-memory advantage on the bulk of the cache.

**Hypothesis:** at decode time, attention mass concentrates on the most
recent W tokens (window). If those W tokens stay in BFP8 (full precision)
and only positions older than W use TQ (quantized), the dominant attention
contributions are accurate and the small contributions from old positions
absorb the quantization noise — recovering most of the lost top-1 without
giving up memory.

**Memory cost analysis** (Llama-3.1-8B, 32 layers, 8 KV heads, head_dim=128):
- Window W=64 tokens × 32 layers × 8 heads × 128 dim × 2 (K+V) × 1 byte (BFP8)
  = ~4 MB total — negligible vs the multi-GB TQ cache at long context.
- TQ cache for the *remaining* positions stays at 0.5 byte/coord (idx) +
  the BF16-padded norms. Memory ratio vs BFP8 baseline ≈ 0.76× (same as today).

---

## Architecture

```
┌─────────────────────────────────────┐
│  attention.py: K, V from QKV projection
└─────────────────────────────────────┘
              │
              ▼ (every step)
┌─────────────────────────────────────┐
│  update_cache(K, V, cur_pos):
│    1. Write to BFP8 ring at slot = cur_pos % W
│    2. Write to TQ cache at slot = cur_pos
│       (whether or not it's still in the ring)
└─────────────────────────────────────┘
              │
              ▼ (decode SDPA call)
┌─────────────────────────────────────┐
│  fused_sdpa_decode_hybrid(Q, cur_pos):
│    if cur_pos < W:
│      # everything's in the ring; pure standard SDPA
│      out = standard_sdpa(Q, BFP8_ring[0..cur_pos], ...)
│    else:
│      # old positions in TQ, recent W in BFP8 ring
│      out_old = fused_TQ_sdpa(Q, TQ_cache[0..cur_pos-W],   ...)
│      out_new = standard_sdpa(Q, BFP8_ring,                ...)
│      out     = combine(out_old, out_new)         # online softmax
│    return out
└─────────────────────────────────────┘
```

Both SDPA calls return `(output, max, sum)` to allow the online softmax
combine. We already have this code from Tier 2A (the cross-core reducer).

---

## Choosing W

Llama attention has a fast-decay tail: the top-K attention weights at any
position usually sit on the **last 32–128 tokens** plus a few "anchor"
tokens (BOS, attention-sinks, etc.). A reasonable starting W is **64**.

Sweep candidates: 32, 64, 128, 256. Larger W = better quality, more memory
+ more BFP8-SDPA cost. Sweet spot likely 64–128 based on attention-sink
papers (Xiao et al. 2024).

Note: the first ~4 tokens (attention sinks) deserve special handling —
keeping them in BFP8 too may help. Easy extension: keep `[0..3]` AND
`[cur_pos-W+1..cur_pos]` in BFP8.

---

## Implementation steps

### Step 1 — Cache structure (`turbo_quant/ttnn_integration.py`)

Add to `TTNNTurboQuantCache`:

```python
def __init__(self, ..., recent_window: int = 0, ...):
    # ... existing TQ cache allocations ...
    self.recent_window = recent_window
    if recent_window > 0:
        # BFP8 ring [num_layers, max_batch, n_kv_heads, W, head_dim]
        # Padded to TILE_LAYOUT for SDPA compatibility.
        self.k_ring_dev = [allocate_bfp8_ring(...) for _ in range(num_layers)]
        self.v_ring_dev = [allocate_bfp8_ring(...) for _ in range(num_layers)]
```

### Step 2 — `update_cache` dual-write

```python
def update_cache(self, k, v, layer_idx, current_pos, page_table):
    # Existing: write to TQ cache at cur_pos
    self._update_tq_cache(k, v, layer_idx, current_pos, page_table)
    # New: write to BFP8 ring at slot = cur_pos % W
    if self.recent_window > 0:
        ring_slot = current_pos % self.recent_window
        # Reuse paged_update_cache with a small page table targeting ring_slot
        ttnn.experimental.paged_update_cache(
            self.k_ring_dev[layer_idx], k, ring_slot, ring_page_table,
        )
        # Same for V
```

### Step 3 — `fused_sdpa_decode_hybrid` in attention.py

```python
def hybrid_sdpa_decode(self, Q, layer_idx, current_pos, scale, page_table):
    cur_pos = ...  # extract scalar
    if cur_pos < self.recent_window:
        # Trivial: all positions in the ring, no TQ at all
        out = standard_sdpa(Q, k_ring, v_ring, scale=scale)
    else:
        # Two SDPA calls, then online softmax combine.
        # The TQ kernel needs to be told to STOP at cur_pos - W instead of cur_pos.
        out_old, max_old, sum_old = fused_tq_sdpa_partial(
            Q, layer_idx, end_pos=cur_pos - self.recent_window, ...,
            return_softmax_state=True,
        )
        out_new, max_new, sum_new = standard_sdpa_partial(
            Q, k_ring, v_ring, scale, return_softmax_state=True,
        )
        out = online_softmax_combine(out_old, max_old, sum_old,
                                     out_new, max_new, sum_new)
    return out
```

The Tier 2A reducer already has this combine math (in
`sdpa_tq_decode.cpp` reducer path). We can either:
  - reuse the kernel as a subroutine (run two SDPA programs and combine
    their outputs at host level), or
  - extend the kernel to read from two cache tensors (more invasive).

Start with **host-level combine** for the first iteration — simpler,
adds one cross-program output round-trip per layer (~1 µs each) but
totally avoids kernel changes. Optimize to fused if perf demands it.

### Step 4 — `eval_e2e.py` and `eval_token_accuracy.py` flags

Add `--tq-recent-window N` (default 0 = disabled, current behavior).
When set, allocate the BFP8 ring and route through hybrid_sdpa_decode.

### Step 5 — Validation

1. **Correctness**: run `eval_e2e.py --tq-full-dequant --tq-recent-window 64`
   on the "Paris" prompt — expect identical answer to baseline TQ FD.
2. **Quality**: run `eval_token_accuracy.py --tq-full-dequant
   --tq-recent-window 64`. Expect top-1 to climb from 86.9 % toward
   97.3 % (BFP8 baseline).
3. **Memory**: confirm `bench_seqlen_sweep.py` shows the TQ KV total
   plus a small constant ring overhead (not scaling with seqlen).
4. **Sweep W**: run with W ∈ {0, 32, 64, 128, 256} to find the sweet
   spot of accuracy vs memory + per-call latency.

---

## Risks & open questions

- **Ring-buffer order**: positions are written modulo W, so the ring
  isn't sequentially ordered. Standard SDPA expects K ordered by
  position. Need to either:
  (a) re-sort the ring at SDPA time (extra ~1 µs op), or
  (b) carry a position offset and fix up the rope handling (rope is
      already absorbed into Q/K via `apply_rotary`, so positions in
      the ring don't need re-rope; just need the attention mask to
      reflect actual positions).

  Likely (b) — rope is per-position invariant; SDPA score is
  Q · K_rope_at_pos, and if the ring stores K_rope_at_pos, the
  out-of-order layout still produces correct scores.

- **Cache initialization**: at decode start (cur_pos=0), the ring is
  empty — guard against reading uninitialized values.

- **Tier 2A interaction**: the existing K=14 cross-core split in the TQ
  kernel still applies to the "old positions" SDPA call. Should compose
  cleanly. The new BFP8-ring SDPA call doesn't use Tier 2A (already fast).

- **Trace capture**: the conditional in `hybrid_sdpa_decode` (cur_pos < W)
  branches at runtime. TTNN trace requires a fixed program. Either:
  (a) always run both SDPA calls and just zero out the TQ result when
      cur_pos < W (extra cost at small cur_pos but uniform program), or
  (b) capture two separate traces and switch at host level (simpler at
      eval level, complicated for serving).

  For the experimental harness, (a) is simpler.

---

## Estimated effort

| Step | Effort |
|---|---|
| Cache structure | ~2 hr |
| update_cache dual-write | ~3 hr |
| hybrid_sdpa_decode + host-level combine | ~4 hr |
| eval_e2e + eval_token_accuracy flag plumbing | ~1 hr |
| Validation runs (correctness, quality, sweep) | ~2 hr |
| **Total** | **~12 hr (~1.5 days)** |

If quality recovers but the host-level combine is too slow, add ~1 day for
fused-kernel combine.

---

## Success criteria

- Top-1 accuracy at W=64 ≥ 95 % (within 2 pp of BFP8 baseline 97.3 %).
- KV memory at long context still ≤ 0.80 × BFP8 baseline.
- Per-call SDPA latency at long context ≤ 1.20× of pure Track A
  (the BFP8-ring SDPA is fast; the TQ SDPA shrinks because it covers
  fewer positions).

If criteria are met → ship hybrid as the production TQ mode.
If top-1 stays < 92 % → the lost positions matter more than expected;
revisit window strategy (longer W or per-position importance ranking).
