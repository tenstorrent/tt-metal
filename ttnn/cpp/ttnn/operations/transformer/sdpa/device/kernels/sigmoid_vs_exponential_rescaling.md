# Sigmoid vs Exponential Rescaling in SDPA Finalization

## Context

File: `compute_common.hpp` lines ~1594–1798
Both standard and ring SDPA share the same inner K-chunk loop (online softmax / flash attention).
They diverge at **finalization** — what happens after all local K chunks are processed.

## Background: Online Softmax (Flash Attention Inner Loop)

Standard flash attention processes the K dimension in chunks, maintaining three running
accumulators per query row:

- **running max** `m` — the largest score seen so far (for numerical stability)
- **running sum** `s` — the sum of `exp((score - m) * scale)` across all K chunks
- **running output** `O` — the un-normalized weighted sum `softmax_numerator @ V`

When a new K chunk arrives with a potentially larger max, all prior statistics must be
**rescaled** by `exp((old_max - new_max) * scale)`:

```
correction = exp((m_old - m_new) * scale)
s_new = s_old * correction + s_chunk
O_new = O_old * correction + O_chunk
```

This exponential rescaling is safe within a single device's inner loop because consecutive
K chunks have similar max values — the correction factor stays in a reasonable numerical range.

### Code (inner loop, lines 1637–1661)

```cpp
// correction = exp((prev_max - cur_max) * scale)
sub_exp_block<scale_fp32>(alias_prev_max, alias_cur_max, cb_exp_max_diff, Sq_chunk_t);

// prev_sum *= correction
mul_tiles_bcast_cols_inplace(alias_prev_sum, cb_exp_max_diff, Sq_chunk_t);

// cur_sum += prev_sum
add_block_inplace(alias_cur_sum, alias_prev_sum, Sq_chunk_t);

// cur_out += prev_out * correction  (L1 accumulation)
mul_block_bcast_cols<Sq_chunk_t, vDHt, false, true>(
    alias_mm2_prev_out, cb_exp_max_diff, alias_mm2_cur_out);
```

After the loop, `alias_prev_sum` holds the un-reduced sum and `alias_mm2_prev_out` holds
the un-normalized output. A final `matmul_reduce` collapses the partial row-sums into
per-row scalars (line 1674).

---

## Standard SDPA Finalization (`else` block, lines 1785–1795)

```cpp
recip_block_inplace(alias_prev_sum, Sq_chunk_t);            // 1/sum
mul_block_bcast_cols(alias_mm2_prev_out, alias_prev_sum, cb_out);  // out /= sum
cb_pop_front(alias_prev_max, Sq_chunk_t);                   // discard max
```

A single device has seen **all** K chunks. Its running max and sum are globally correct.
Finalization is just dividing the output by the sum. The max is discarded — it served its
purpose during the loop and is never needed again.

**No LSE, no cross-device merge, no sigmoid.**

---

## Ring SDPA Finalization (`if constexpr (sdpa_type == RING)`, lines 1724–1784)

Ring attention distributes the K dimension across **multiple devices** in a ring topology.
Each device sees only a **subset** of K chunks, so after the inner loop its max/sum/output
are only *locally* correct. Two additional steps are needed:

### Step 1: Compute local LSE and normalize locally (lines 1725–1734)

```cpp
log_block(alias_prev_sum, alias_cur_max, Sq_chunk_t);       // cur_max = log(sum)
mul_block_bcast_scalar_inplace<cb_scale_in>(alias_prev_max); // prev_max *= scale
add_block_inplace(alias_prev_max, alias_cur_max, Sq_chunk_t); // LSE = scale*max + log(sum)

recip_block_inplace(alias_prev_sum, Sq_chunk_t);             // 1/sum
mul_block_bcast_cols_inplace(alias_mm2_prev_out, alias_prev_sum); // out /= sum
```

**Why LSE?** The log-sum-exp statistic compactly represents the softmax denominator:

```
LSE = log(Σ exp(x_i * scale))
    = scale * max + log(Σ exp((x_i - max) * scale))
    = scale * max + log(sum)
```

Standard SDPA discards the max because the answer is final. Ring SDPA **must** preserve it
(as LSE) because it's the key needed to correctly merge partial results across devices.

The output is also locally normalized (divided by sum), making it a proper weighted average
for *this device's* K-chunk subset. This is different from standard SDPA where the output
is only normalized once at the very end — ring SDPA normalizes locally so that the merge
step can work with properly-scaled partial outputs.

### Step 2: Merge with previous device's result via sigmoid (lines 1736–1777)

On `ring_iter > 0`, this device receives `prev_out` and `prev_lse` from the previous
device in the ring. The merge formulas are:

```
sig  = sigmoid(cur_lse - prev_lse)
out  = prev_out - sig * (prev_out - cur_out)
lse  = prev_lse - log_sigmoid(prev_lse - cur_lse)
```

#### Code (lines 1751–1777)

```cpp
// sig = sigmoid(cur_lse - prev_lse)
sigmoid_sub(alias_cur_lse, cb_lse_in, alias_sig, Sq_chunk_t);

// sub = prev_out - cur_out
sub_block(cb_prev_out, alias_cur_out, alias_sub, out_chunk_tiles);
// sub *= sig  (broadcast cols since sig is per-row)
mul_block_bcast_cols_inplace<Sq_chunk_t, DHt>(alias_sub, alias_sig);
// out = prev_out - sub
sub_block(cb_prev_out, alias_sub, cb_out, out_chunk_tiles);

// LSE update: lse = prev_lse - log_sigmoid(prev_lse - cur_lse)
logsigmoid_sub(cb_lse_in, alias_cur_lse, alias_sig, Sq_chunk_t);
sub_block(cb_lse_in, alias_sig, cb_lse_out, Sq_chunk_t);
```

On `ring_iter == 0` (first iteration, no previous device result yet), the local output
and LSE are simply copied to the output CBs (lines 1778–1784).

---

## Why Sigmoid Instead of Exponential Rescaling?

### The naive approach (what the inner loop does)

The textbook way to combine two partial softmax results with LSE values `L1` and `L2`:

```
weight1 = exp(L1) / (exp(L1) + exp(L2))
weight2 = exp(L2) / (exp(L1) + exp(L2))
out = weight1 * out1 + weight2 * out2
```

This requires computing `exp(L1)` and `exp(L2)` directly, which **overflows** when LSE
values are large (they grow with sequence length and score magnitudes).

The inner loop gets away with exponential rescaling because it computes `exp(old_max - new_max)`,
and consecutive chunks on the same device have similar max values. The difference is bounded.

### The problem across devices

Across devices in a ring, LSE values can differ **dramatically**. One device might process
the most-attended K tokens (high LSE), while another sees irrelevant tokens (low LSE).
The difference `L1 - L2` can be arbitrarily large, making `exp(L1 - L2)` overflow or
`exp(L2 - L1)` underflow.

### The sigmoid reformulation

Rewriting the weights using the sigmoid function:

```
weight1 = exp(L1) / (exp(L1) + exp(L2))
        = 1 / (1 + exp(L2 - L1))
        = sigmoid(L1 - L2)
```

Key properties:
- **sigmoid(x)** is always in `[0, 1]`, regardless of how extreme the LSE difference is
- No raw exponentials of large values are ever computed
- The output formula `prev_out - sig * (prev_out - cur_out)` is a **numerically stable
  convex interpolation**:
  - When `cur_lse >> prev_lse`: `sig -> 1`, result approaches `cur_out`
  - When `cur_lse << prev_lse`: `sig -> 0`, result approaches `prev_out`

### The LSE update

Similarly, combining LSE values:

```
new_lse = log(exp(L1) + exp(L2))
        = L1 + log(1 + exp(L2 - L1))
        = L1 - log_sigmoid(L1 - L2)
```

`log_sigmoid` avoids computing large exponentials. In code:

```cpp
logsigmoid_sub(cb_lse_in, alias_cur_lse, alias_sig, Sq_chunk_t);  // log_sigmoid(prev - cur)
sub_block(cb_lse_in, alias_sig, cb_lse_out, Sq_chunk_t);          // prev - log_sigmoid(...)
```

---

## Mathematical Equivalence Proof

Both approaches compute the same result. Starting from the weighted combination:

```
out = (exp(L1) * O1 + exp(L2) * O2) / (exp(L1) + exp(L2))
```

Let `sig = sigmoid(L2 - L1) = exp(L2) / (exp(L1) + exp(L2))`, so `1 - sig = sigmoid(L1 - L2)`.

```
out = (1 - sig) * O1 + sig * O2
    = O1 - sig * (O1 - O2)
```

Which matches the code: `out = prev_out - sig * (prev_out - cur_out)`.

For LSE:

```
new_lse = log(exp(L1) + exp(L2))
        = L1 + log(1 + exp(L2 - L1))
        = L1 - log(sigmoid(L1 - L2))       [since log(1+e^x) = -log(sigmoid(-x))]
        = L1 - log_sigmoid(L1 - L2)
```

Which matches: `lse = prev_lse - log_sigmoid(prev_lse - cur_lse)`.

---

## Summary Table

| Aspect                   | Standard SDPA (`else`)             | Ring SDPA                                       |
|--------------------------|------------------------------------|-------------------------------------------------|
| K coverage               | All chunks on one device           | Subset per device                               |
| Max/Sum after loop       | Globally correct                   | Locally correct only                            |
| Finalization             | Simple `out /= sum`               | Compute LSE, normalize, merge with ring neighbor|
| LSE needed?              | No — max discarded                 | Yes — exported for cross-device merge           |
| Rescaling method         | `exp(old_max - new_max)` (bounded) | `sigmoid(lse_a - lse_b)` (always in [0,1])     |
| Merge stability          | N/A (no merge)                     | Sigmoid-based, avoids overflow across devices   |
| Extra inputs             | None                               | `cb_lse_in`, `cb_prev_out` from ring neighbor   |
| Extra outputs            | `cb_out` only                      | `cb_out` + `cb_lse_out` (forwarded in ring)     |
| Output normalization     | Once at end                        | Locally per device, then merge adjusts weights  |

---

## Risk Analysis: Exponential Rescaling Across the Entire Ring

**Proposed alternative**: Skip the sigmoid-based merge. Instead, keep doing exponential
rescaling (`exp(old_max - new_max)`) across ring iterations — treating cross-device chunks
the same way as intra-device chunks — and divide by the accumulated sum only at the end.

### Risk 1: Overflow in the correction factor (primary risk)

The inner K-chunk loop computes `exp(old_max - new_max)`. This works because consecutive
chunks on the **same device** have similar max values — the difference is bounded.

Across ring devices, this guarantee **disappears**. Consider causal or sparse attention:
- Device 0 processes the most-attended K tokens → max = 50
- Device 3 processes irrelevant tokens → max = -10

The correction `exp(50 - (-10)) = exp(60)` ≈ 1.14×10^26. In bfloat16 (max ≈ 3.4×10^38)
this *might* fit, but in fp16 (max ≈ 65504) it's instant overflow. Chain a few of these
across 8+ ring steps and even fp32 accumulation becomes risky.

### Risk 2: Multiplicative error accumulation

Each ring step would apply another multiplicative rescaling to the **entire running output**
(Sq_chunk_t × vDHt tiles). Over N ring iterations, you get N multiplicative corrections,
each introducing rounding error. These compound — after 8 ring steps you've multiplied the
output by a chain of correction factors, each rounded in bfloat16/fp16b.

The sigmoid approach does a **single convex interpolation** per ring step
(`prev_out - sig * (prev_out - cur_out)`), bounded in [0,1]. No compounding.

### Risk 3: Running sum overflow

Without local normalization, the running sum `s = s_old * correction + s_chunk` grows with
each ring device. The sum tracks `Σ exp(score - global_max)` across all devices. For long
sequences across many devices, this can become very large before the final division.

### Risk 4: Output magnitude blows up

Currently, each device normalizes its output (`out /= sum`) before the merge, keeping
values in a reasonable range. If you skip this and carry raw unnormalized output across the
ring, intermediate values scale with the sum of exponentials — potentially orders of
magnitude larger than the final result. This wastes dynamic range in low-precision formats
and increases rounding error in the final division.

### When it *could* work

The approach would be safe if:
- All devices see similar max values (uniform attention, no causal masking)
- fp32 accumulation is used throughout the ring merge
- Ring size is small (2–4 devices)

But these conditions can't be guaranteed for general workloads (causal masks,
variable-length sequences, MQA/GQA with heterogeneous attention patterns).

### Bottom line

The sigmoid reformulation exists precisely because the cross-device LSE differences are
**unbounded** in the general case. Keeping exponential rescaling trades ~6 extra tile ops
per ring step (sigmoid, logsigmoid, sub, mul) for **guaranteed numerical stability**
regardless of attention pattern, sequence length, or ring size. The performance cost is
modest; the numerical risk of removing it is workload-dependent and hard to bound.

---

## Knowledge Sources

- [GPU MODE Lecture 13: Ring Attention](https://christianjmills.com/posts/cuda-mode-notes/lecture-013/) — walkthrough of ring attention mechanics
- [Ring Attention LSE notebook (gpu-mode)](https://github.com/gpu-mode/ring-attention/blob/main/notebooks/howto_log_sum_exp.ipynb) — derivation of sigmoid-based LSE combination, shows why `1/(1+exp(L2-L1))` replaces direct exponentials
- [From Online Softmax to FlashAttention (UW CSE599m)](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf) — the online softmax algorithm and rescaling step
- [Ring Attention Explained (Coconut Mode)](https://coconut-mode.com/posts/ring-attention/) — high-level ring attention overview with cumulative normalization
- [Ring Attention Explained (Akasa)](https://akasa.com/blog/ring-attention) — ring topology and KV chunk distribution
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) — numerical stability fundamentals
- [FLASH-D: FlashAttention with Hidden Softmax Division (arXiv 2505.14201)](https://arxiv.org/abs/2505.14201) — alternative softmax division strategies
