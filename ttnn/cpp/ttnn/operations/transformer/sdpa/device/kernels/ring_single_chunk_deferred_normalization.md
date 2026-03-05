# Ring SDPA: Deferred Normalization for Single Q-Chunk

## Problem Statement

The current ring SDPA implementation normalizes and merges output **per Q chunk per ring
iteration**. This involves:

1. **Compute overhead** per ring iteration per Q chunk:
   - LSE computation: `log(sum)`, `scale * max + log(sum)`
   - Local normalization: `recip(sum)`, `out *= 1/sum`
   - Sigmoid merge: `sigmoid(cur_lse - prev_lse)`, interpolation, LSE update via
     `logsigmoid`

2. **DRAM bandwidth overhead** per ring iteration per Q chunk:
   - Writer writes merged output + LSE to DRAM (`write_output_and_lse`)
   - On the next ring iteration, writer reads them back (`read_prev_output_and_lse`)

This is visible in `ring_joint_writer.cpp` lines 231–263:

```cpp
for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
    ...
    for (uint32_t global_q_chunk = global_q_start; ...) {
        if (ring_iter > 0) {
            read_prev_output_and_lse(..., cb_prev_out, cb_lse_in, ...);  // DRAM read
        }
        // compute kernel: sigmoid merge → cb_out, cb_lse_out
        write_output_and_lse(..., cb_out, cb_lse_out, ...);              // DRAM write
    }
    noc_async_write_barrier();
}
```

When there is only **one Q chunk** per core, these costs are avoidable.

---

## Why Multiple Q Chunks Force Per-Iteration Finalization

KV data arrives from the ring in a fixed order — all Q chunks must process the current ring
iteration's KV before the next iteration's KV arrives (the data is transient, forwarded
through the ring). This forces an outer-ring / inner-Q loop order:

```
for each ring_iter:          // KV arrives in ring order
  for each q_chunk:          // must process all Q against current KV
    process local K chunks
    finalize: normalize + sigmoid merge
    write merged output + LSE to DRAM    ← frees L1 for next Q chunk
```

Between ring iterations, each Q chunk's merged output must persist **somewhere**. Since L1
(1.5MB) cannot hold all Q chunks' intermediate state simultaneously, DRAM is the only
option. The DRAM round-trip (write after finalize, read before next merge) is unavoidable.

---

## Proposed Optimization: Single Q-Chunk Deferred Normalization

When there is exactly **one Q chunk** per core, only one set of accumulators is needed:
- Running output: `Sq_chunk_t × vDHt` tiles
- Running max: `Sq_chunk_t` tiles
- Running sum: `Sq_chunk_t` tiles

These fit comfortably in L1 across all ring iterations. The optimization:

### Before (current flow)

```
for each ring_iter:
  for q (just 1):
    for each k_chunk:
      exponential rescaling (inner loop)          ← same as standard SDPA
    normalize: out /= sum
    compute LSE = scale * max + log(sum)
    sigmoid merge with prev_out / prev_lse        ← ~6 tile ops
    write merged output + LSE to DRAM             ← DRAM write
    (next iter: read back from DRAM)              ← DRAM read
```

### After (proposed flow)

```
for q (just 1):
  for each ring_iter:
    for each k_chunk:
      exponential rescaling (inner loop)          ← identical to intra-device path
  normalize: out /= sum                           ← once, at the very end
  write final output to DRAM                      ← single DRAM write
```

All ring iterations collapse into one continuous stream of K chunks. The compute kernel
treats cross-device KV chunks identically to intra-device ones, using the same
`exp(old_max - new_max)` rescaling that the inner K-chunk loop already does.

### What is eliminated

| Per-iteration cost removed       | Tiles / ops saved per ring step           |
|----------------------------------|-------------------------------------------|
| LSE computation                  | log + mul_scalar + add (3 ops × Sq_chunk_t tiles) |
| Local normalization              | recip + mul_bcast_cols (2 ops × Sq_chunk_t + vDHt tiles) |
| Sigmoid merge                    | sigmoid_sub + sub + mul_bcast_cols + sub (4 ops × out_chunk_tiles) |
| LSE update                       | logsigmoid_sub + sub (2 ops × Sq_chunk_t tiles) |
| DRAM write (output + LSE)        | (Sq_chunk_t × vDHt + Sq_chunk_t) tiles × tile_bytes |
| DRAM read (prev_out + prev_LSE)  | Same as above                             |

With `ring_size = 8`, this saves 7 iterations worth of the above (first iteration has no
merge in either approach).

---

## Numerical Stability Risks

The core risk is that exponential rescaling across devices is **not bounded** the way it is
within a single device's K-chunk loop. See
[sigmoid_vs_exponential_rescaling.md](./sigmoid_vs_exponential_rescaling.md) for the full
analysis. Summary:

### Risk 1: Correction factor overflow

Within a device, consecutive K chunks have similar max values, so `exp(old_max - new_max)`
stays in a reasonable range. Across devices, max values can differ dramatically (e.g.,
causal masking: one device sees heavily-attended tokens, another sees irrelevant ones).

Example: device 0 max = 50, device 3 max = −10 → `exp(60)` ≈ 1.14×10^26. Overflows fp16
(max ≈ 65504). In bfloat16 it fits but is near the edge; chain several such corrections
across 8+ ring steps and even fp32 is at risk.

### Risk 2: Multiplicative error accumulation

Each ring step applies another multiplicative correction to the entire running output. Over
N ring iterations, N corrections compound, each rounded in bfloat16/fp16b. The sigmoid
approach avoids this — it does a single bounded convex interpolation per step.

### Risk 3: Running sum overflow

Without local normalization, the sum `s = s_old × correction + s_chunk` grows across all
devices. For long sequences across many devices, this can overflow before the final
division.

### Risk 4: Unnormalized output magnitude

Skipping local normalization means intermediate output values scale with the sum of
exponentials, wasting dynamic range in low-precision formats and increasing rounding error.

### When the risks are acceptable

- All devices see similar max values (uniform attention, no causal masking)
- fp32 accumulation used throughout
- Small ring size (2–4 devices)
- Workload-specific validation confirms acceptable PCC

---

## Implementation Considerations

### Compute kernel changes (`compute_streaming.hpp`)

The `sdpa_ring_v2` function (line 987) would need a new code path (e.g., gated by a
compile-time flag like `single_q_chunk_deferred_norm`) that:

1. Removes the per-Q finalization block (lines 1097–1153) from inside the ring iteration
   loop
2. Moves the K-chunk processing loop to span all ring iterations for the single Q chunk
3. Adds a single finalization after all ring iterations:
   - `matmul_reduce` for row-reducing partial sums
   - `recip_block_inplace` + `mul_block_bcast_cols` for `out /= sum`
4. Removes `cb_lse_in`, `cb_lse_out`, `cb_prev_out` dependencies (no longer needed)

### Writer kernel changes (`ring_joint_writer.cpp`)

- Remove `read_prev_output_and_lse` / `write_output_and_lse` per-iteration calls
- Write final output to DRAM once after the ring loop completes
- LSE output can still be computed at the end if needed downstream (for multi-device
  aggregation at a higher level)

### Program factory changes

- Gate the optimization on `num_q_chunks == 1` (single Q chunk per core)
- Adjust CB allocations: `cb_lse_in`, `cb_lse_out`, `cb_prev_out` can be removed or
  reduced, freeing L1 for other uses
- Ensure the accumulator CBs (`cb_out_im_A/B`, `cb_max_A/B`, `cb_sum_A/B`) persist across
  ring iterations (currently they are reused per Q chunk, which already works for single Q)

### Validation

- Compare PCC against the sigmoid-based path for various attention patterns (uniform,
  causal, sparse, variable-length)
- Test with different ring sizes (2, 4, 8, 32)
- Test with different data formats (fp16, bfloat16, fp32 accumulation)
- Monitor intermediate accumulator magnitudes to detect overflow-prone configs

---

## Knowledge Sources

- [sigmoid_vs_exponential_rescaling.md](./sigmoid_vs_exponential_rescaling.md) — detailed
  analysis of why sigmoid-based merging exists and the numerical risks of exponential
  rescaling across devices
- [FLASH-D: FlashAttention with Hidden Softmax Division (arXiv 2505.14201)](https://arxiv.org/abs/2505.14201) — alternative
  softmax division strategies relevant to deferred normalization
- [GPU MODE Lecture 13: Ring Attention](https://christianjmills.com/posts/cuda-mode-notes/lecture-013/) — ring
  attention mechanics and cross-device merge strategies
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) — numerical
  stability fundamentals for softmax computation
