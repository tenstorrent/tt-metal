# Zigzag Work Distribution Plan

## Context

Profiling revealed **H4: Work Imbalance** as the #1 bottleneck (see `zigzag_profile_analysis.md`):
- 116x imbalance between fastest (1.2ms) and slowest (139ms) cores
- 36 cores idle 85%+ of time, 46 cores gate wall-clock
- Potential 2x speedup from rebalancing alone

**Root cause:** Current algorithm distributes Q chunks uniformly by COUNT, ignoring causal masking work variance where Q chunk 0 processes ~3 K chunks vs Q chunk 15 processes ~32 K chunks.

---

## Current Algorithm

**Location:** `ring_joint_sdpa_profile_program_factory.cpp:578-641`

```cpp
const uint32_t total_q_chunks = B * NH * num_q_chunks;
const uint32_t base_chunks_per_core = total_q_chunks / num_cores;
const uint32_t extra_chunks = total_q_chunks % num_cores;

for (uint32_t i = 0; i < num_cores; ++i) {
    uint32_t chunk_count = base_chunks_per_core + ((i < extra_chunks) ? 1 : 0);
    uint32_t global_q_start = next_global_chunk;
    uint32_t global_q_end = next_global_chunk + chunk_count;
    next_global_chunk += chunk_count;
}
```

**Problem:** Contiguous block assignment. Core 0 gets Q chunks [0, 5), which are ALL early chunks with minimal causal work. Core 109 gets Q chunks [507, 512), which are ALL late chunks with maximum causal work.

---

## Alternative Solutions Considered

### Option A: Interleaved (Strided) Assignment

Assign Q chunks in round-robin across cores so each core gets a mix of early and late chunks.

**Pros:** Simple, naturally balances causal work
**Cons:** Doesn't perfectly balance, changes memory access patterns

### Option B: Work-Weighted Greedy Assignment

Calculate expected K chunks per Q chunk, then assign greedily to least-loaded core.

**Pros:** Optimal balance, handles arbitrary work distributions
**Cons:** More complex, non-contiguous assignment complicates kernel

### Option C: Two-Group Split (Light/Heavy)

Split Q chunks into light (early) and heavy (late) groups, distribute each group separately.

**Pros:** Simpler than full greedy
**Cons:** Specific to causal pattern, may not generalize

---

## Recommended: Zigzag (Mirrored) Assignment

**Idea:** Pair each light Q chunk with its mirror-heavy counterpart. Core i gets chunks {i, n-1-i} creating natural load balance.

```
Q chunk work (causal, 16 chunks, 32 K chunks total):
  q0:  3 K chunks  <--+--> q15: 32 K chunks  = 35 total
  q1:  5 K chunks  <--+--> q14: 30 K chunks  = 35 total
  q2:  7 K chunks  <--+--> q13: 28 K chunks  = 35 total
  q3:  9 K chunks  <--+--> q12: 26 K chunks  = 35 total
  ...

Zigzag assignment:
  Core 0: [q0, q15]  -> 35 K chunks
  Core 1: [q1, q14]  -> 35 K chunks
  Core 2: [q2, q13]  -> 35 K chunks
  ...
  Core 7: [q7, q8]   -> 34 K chunks
```

**Mathematical guarantee:** For causal attention, work(qi) + work(q_{n-1-i}) = constant.

**Pros:**
- **Perfect balance** for causal masking (mathematical property)
- Simple to implement (just pair chunks)
- **Minimal kernel changes** - can still use contiguous pairs
- No runtime overhead
- Generalizes to any causal pattern

**Cons:**
- Requires even number of Q chunks (or handle middle chunk specially)
- Specific to causal masking (but that's our use case)

**Odd number of chunks:** If total_q_chunks is odd, one core handles the middle chunk alone and does half the work. This is an acceptable tradeoff.

---

## Implementation Strategy

### Host (program_factory)

```cpp
// Instead of contiguous Q chunk ranges, assign pair indices
// Each core processes pairs [pair_start, pair_end)
// A "pair" is (q_lo, q_hi) where q_hi = total_q_chunks - 1 - q_lo

uint32_t num_pairs = total_q_chunks / 2;
uint32_t pairs_per_core = num_pairs / num_cores;
uint32_t extra_pairs = num_pairs % num_cores;

for (uint32_t i = 0; i < num_cores; ++i) {
    uint32_t my_pair_count = pairs_per_core + ((i < extra_pairs) ? 1 : 0);
    // Set runtime args: pair_start, pair_count, total_q_chunks
}

// Handle odd total_q_chunks: one core also processes middle chunk
if (total_q_chunks % 2 == 1) {
    uint32_t middle_chunk = total_q_chunks / 2;
    // Assign to last core or distribute
}
```

### Kernel (reader/writer/compute)

```cpp
// Replace: for (q = global_q_start; q < global_q_end; ++q)
// With:
for (uint32_t pair = pair_start; pair < pair_start + pair_count; ++pair) {
    uint32_t q_lo = pair;
    uint32_t q_hi = total_q_chunks - 1 - pair;

    // Process light chunk first, then heavy chunk
    process_q_chunk(q_lo);
    process_q_chunk(q_hi);
}

// Handle middle chunk if assigned (odd total_q_chunks)
if (has_middle_chunk) {
    process_q_chunk(middle_chunk);
}
```

---

## Expected Results

```
Before (contiguous):
  Core 0:   [q0..q4]     -> ~20 K chunks total  (1.2ms)
  Core 109: [q507..q511] -> ~160 K chunks total (139ms)
  Imbalance: 116x

After (zigzag):
  Core 0:   [q0,q511], [q1,q510], ...  -> ~70 K chunks total
  Core 109: [q109,q402], ...           -> ~70 K chunks total
  Imbalance: ~1.2x
```

---

## Files to Modify

| File | Change |
|------|--------|
| `ring_joint_sdpa_profile_program_factory.cpp` | Change chunk assignment from contiguous to zigzag pairs |
| `ring_joint_profile_reader.cpp` | Replace global_q_start/end loop with pair iteration |
| `ring_joint_profile_writer.cpp` | Replace global_q_start/end loop with pair iteration |
| `compute_common.hpp` (sdpa_compute) | Replace global_q_start/end loop with pair iteration |

---

## Verification

1. Run profiled test and compare core duration histogram
2. Expect: imbalance reduced from **116x to ~1.2x** (near-perfect balance)
3. Expect: wall-clock time reduced from **139ms to ~70ms** (2x speedup)
4. Verify PCC still >0.99 (correctness unchanged)

```bash
# Run test with profiling
python -m tracy -p -r -v -m pytest "tests/ttnn/unit_tests/operations/sdpa/test_ring_joint_sdpa_profile.py::test_ring_joint_sdpa_profile_production_scale[ring_index=0]" \
    -v -s

# Compare kernel durations
grep "DEVICE KERNEL DURATION" generated/profiler/reports/<timestamp>/ops_perf_results_*.csv
```
