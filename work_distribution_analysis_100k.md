# Work Distribution Analysis: Ring Joint SDPA (100k Sequence)

## Context
Analysis of work distribution and core utilization for ring_joint_sdpa_profile with causal+balanced mode on 100k sequence.

---

## Configuration

```
Total sequence: 102,400 tokens (~100k)
Ring size: 32 devices
is_balanced: True (zigzag device assignment)
is_causal: True

Device 0 (ring_index=0):
  Global chunks: [0, 63] (zigzag assignment)
  Token ranges: [0-1,599] + [100,800-102,399] = 3,200 local tokens

q_chunk_size = 160 tokens
k_chunk_size = 160 tokens
num_local_q_chunks = 3,200 / 160 = 20 Q chunks per head
num_heads = 32
total_q_chunks = 32 * 20 = 640
num_cores = 110
```

---

## Work Per Q Chunk

For causal+balanced mode with ring_index=0:
- **Light Q chunks (q0-q9)**: Process only local KV with causal mask (early sequence)
- **Heavy Q chunks (q10-q19)**: Process ALL KV from ALL 32 devices (late sequence)

```
Q Chunk | Global Positions     | Work (K chunks to process)
--------|----------------------|---------------------------
   0    | [0-159]              |     1   (local causal)
   1    | [160-319]            |     2   (local causal)
   2    | [320-479]            |     3   (local causal)
   ...  | ...                  |   ...
   9    | [1,440-1,599]        |    10   (local causal)
--------|----------------------|---------------------------
  10    | [100,800-100,959]    |   640   (ALL 32 devices)
  11    | [100,960-101,119]    |   640   (ALL 32 devices)
  ...   | ...                  |   ...
  19    | [102,240-102,399]    |   640   (ALL 32 devices)

Work ratio: Heavy (640) vs Light avg (5.5) = ~116x
```

---

## Current Q Chunk Assignment

From `ring_joint_sdpa_profile_program_factory.cpp` (lines 579-594):

```cpp
const uint32_t total_q_chunks = B * NH * num_q_chunks;  // 640
uint32_t next_global_chunk = 0;

for (uint32_t i = 0; i < num_cores; ++i) {
    uint32_t global_q_start = next_global_chunk;
    // Sequential assignment - cores get consecutive Q chunks
```

Q chunks are assigned **sequentially** by `(batch, head, q_chunk_idx)`:
- Q chunk 0 = (b=0, h=0, q=0)  -> Core 0
- Q chunk 1 = (b=0, h=0, q=1)  -> Core 0
- ...
- Q chunk 19 = (b=0, h=0, q=19) -> Core 0 or 1
- Q chunk 20 = (b=0, h=1, q=0)  -> Core 1
- ...

With 640 Q chunks / 110 cores = 5.8 chunks/core:
- **First ~18 cores**: Get Q chunks from early heads, predominantly light (q0-q9)
- **Last ~18 cores**: Get Q chunks from late heads, predominantly heavy (q10-q19)
- **Middle cores**: Mixed

---

## Measured Results (Tracy Profiling)

### Test Configuration
```
ring_index=0, total_seq=102400, q_chunk=160, k_chunk=160
Device: Blackhole, 110 compute cores
```

### TRISC_2 Kernel Duration Per Core
```
Min:    1,230,304 cycles (  0.91 ms)  <- Light Q chunks only
Max:  162,408,871 cycles (120.30 ms)  <- Heavy Q chunks
Avg:   64,094,909 cycles ( 47.48 ms)

MEASURED UTILIZATION (avg/max): 39.5%
```

### Work Distribution by Core
```
< 5M cycles   (< 3.7ms):   21 cores  <- Got mostly light Q chunks
5-20M cycles  (3.7-15ms):  12 cores
20-50M cycles (15-37ms):   14 cores
50-100M cycles (37-74ms):  30 cores
100-150M cycles (74-111ms): 30 cores
>150M cycles  (>111ms):     3 cores  <- Got mostly heavy Q chunks
```

### Imbalance Metrics
```
Total compute work: 7,050,440,004 cycles (5.22 seconds aggregate)
Optimal per core:    64,094,909 cycles (47.48 ms)
Actual bottleneck:  162,408,871 cycles (120.30 ms)

Imbalance overhead: 153%
Max/Min ratio: 132x
```

---

## Root Cause

The sequential Q chunk assignment creates work imbalance because:

1. **Q chunk ordering**: (batch, head, q_chunk_idx) means light chunks (q0-q9) come before heavy chunks (q10-q19) within each head

2. **Sequential core assignment**: Core 0 gets chunks 0-5, Core 1 gets chunks 6-11, etc.

3. **Result**: Early cores get predominantly light work, late cores get predominantly heavy work

### Example Assignment (simplified)
```
Core 0:  (h0,q0), (h0,q1), (h0,q2), (h0,q3), (h0,q4), (h0,q5)
         -> Work: 1+2+3+4+5+6 = 21 K chunks (light)

Core 18: (h2,q18), (h2,q19), (h3,q0), (h3,q1), (h3,q2), (h3,q3)
         -> Work: 640+640+1+2+3+4 = 1,290 K chunks (mixed)

Core 100: (h31,q8), (h31,q9), (h31,q10), (h31,q11), (h31,q12), (h31,q13)
          -> Work: 9+10+640+640+640+640 = 2,579 K chunks (heavy)
```

---

## What `is_balanced` Actually Does

The `is_balanced` flag in the compute kernel (`ring_joint_sdpa.cpp` lines 224-227) handles **causal masking** with zigzag device assignment:

```cpp
if (is_causal && is_balanced && ring_index > ring_id) {
    iter_num_kv_chunks /= 2;  // Skip processing remote KV for early Q chunks
}
```

This ensures correctness when a device has both early and late sequence chunks, but it does **NOT** balance work across cores.

---

## Recommendations

### To Achieve Better Utilization

1. **Interleave light and heavy Q chunks in assignment order**:
   ```
   Instead of: q0, q1, q2, ..., q9, q10, q11, ..., q19
   Use:        q0, q19, q1, q18, q2, q17, ...  (zigzag within head)
   ```

2. **Or distribute by work estimate**:
   - Count heavy chunks, distribute evenly first
   - Fill remaining core capacity with light chunks

3. **Change in program factory**:
   - Currently: `global_q_start = next_global_chunk++` (sequential)
   - Needed: Work-aware chunk ordering before assignment

### Expected Improvement

With proper work balancing:
- 320 heavy chunks / 110 cores = 2.91 heavy/core
- Bottleneck: ~3 heavy chunks = 1,920 K chunks
- Optimal: 206,560 K / 110 = 1,878 K chunks
- **Theoretical utilization: 97.8%** (vs current 39.5%)

---

## Summary

| Metric | Current | With Work Balancing |
|--------|---------|---------------------|
| Utilization | 39.5% | ~97.8% |
| Bottleneck | 120.30 ms | ~48.5 ms |
| Max/Min ratio | 132x | ~1.02x |
| Speedup | 1.0x | **2.5x** |
