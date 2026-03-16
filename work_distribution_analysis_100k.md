# Work Distribution Analysis: 100k Sequence with q_chunk=320, k_chunk=160

## Context
Extending the 128k work distribution analysis to a ~100k sequence with different chunking parameters to understand how chunk sizes affect:
1. L1 memory fit
2. Work distribution imbalance
3. Theoretical improvement from balancing

---

## Configuration

### Sequence Parameters (using 102,400 for clean math)
```
Total sequence: 102,400 tokens (≈100k, divisible by 64×32 for alignment)
Ring size: 32 devices
is_balanced: True (zigzag device assignment)
is_causal: True

Global chunks: 64 (at 1,600 tokens each)
Device chunk pairs: 32 (each device gets 2 global chunks from zigzag)

Device 0 (ring_index=0):
  Global chunks: [0, 63]
  Token ranges: [0-1,599] + [100,800-102,399] = 3,200 local tokens

q_chunk_size = 320 tokens → Sq_chunk_t = 10 tiles
k_chunk_size = 160 tokens → Sk_chunk_t = 5 tiles
DHt = 18 tiles (576 dim head / 32)
vDHt = 18 tiles

num_local_q_chunks = 3,200 / 320 = 10 Q chunks
num_local_k_chunks = 3,200 / 160 = 20 K chunks

Parallelization:
  total_q_chunks = B × NH × num_q_chunks = 1 × 32 × 10 = 320
  num_cores = 110
  chunks_per_core: 100 cores get 3 chunks, 10 cores get 2 chunks
```

---

## L1 Memory Analysis

### Circular Buffer Requirements

From `ring_joint_sdpa_program_factory.cpp`:
```cpp
q_tiles = Sq_chunk_t × DHt × q_buffer_factor = 10 × 18 × 2 = 360 tiles
k_tiles = Sk_chunk_t × DHt × 2 = 5 × 18 × 2 = 180 tiles  // double buffer
v_tiles = Sk_chunk_t × vDHt × 2 = 5 × 18 × 2 = 180 tiles  // double buffer
mask_tiles = Sq_chunk_t × Sk_chunk_t = 10 × 5 = 50 tiles
qk_tiles = Sq_chunk_t × Sk_chunk_t = 10 × 5 = 50 tiles
out_im_tiles = Sq_chunk_t × vDHt = 10 × 18 = 180 tiles
out0_t = Sq_chunk_t × vDHt = 10 × 18 = 180 tiles
statistics_tiles = Sq_chunk_t = 10 tiles (× multiple CBs)
```

### Memory Calculation (BFloat16 = 2KB/tile)

| Buffer     | Tiles | Tile Size | Memory (KB) |
|------------|-------|-----------|-------------|
| Q input    | 360   | 2 KB      | 720         |
| K input    | 180   | 2 KB      | 360         |
| V input    | 180   | 2 KB      | 360         |
| Mask       | 50    | 256 B     | 12.5        |
| QK intermed| 50    | 2 KB      | 100         |
| Out intermed (×3) | 540 | 2 KB  | 1,080       |
| Statistics (×6) | 60  | 2 KB    | 120         |
| Output     | 180   | 2 KB      | 360         |
| Scale (×3) | 3     | 2 KB      | 6           |
| **TOTAL**  |       |           | **~3,118 KB** |

### Verdict: Does NOT fit in L1

**L1 capacity: ~1,500 KB per Tensix core**
**Required: ~3,118 KB**
**Overcommit: 2.1×**

The q_chunk_size=320 with k_chunk_size=160 configuration exceeds L1 by over 2×. This would require either:
- Smaller chunk sizes
- Streaming/spilling strategies
- Different data formats

---

## Theoretical Work Distribution (Ignoring L1)

### Q Chunk → Global Position Mapping

Device 0 with is_balanced=True:
```
Local Q chunks 0-4:  → Global tokens [0-1,599]        (chunk 0, early sequence)
Local Q chunks 5-9:  → Global tokens [100,800-102,399] (chunk 63, late sequence)
```

### Work Per Q Chunk

For balanced mode with ring_index=0:
- Q chunks 0-4 (early): **SKIPPED** for remote KV (only local KV with causal mask)
- Q chunks 5-9 (late): Process ALL KV (local + 31 remote devices)

```
Q Chunk │ Global Positions     │ Local KV (causal) │ Remote KV       │ Total K chunks
────────┼──────────────────────┼───────────────────┼─────────────────┼───────────────
   0    │ [0-319]              │       2           │  SKIPPED        │      2
   1    │ [320-639]            │       4           │  SKIPPED        │      4
   2    │ [640-959]            │       6           │  SKIPPED        │      6
   3    │ [960-1,279]          │       8           │  SKIPPED        │      8
   4    │ [1,280-1,599]        │      10           │  SKIPPED        │     10
────────┼──────────────────────┼───────────────────┼─────────────────┼───────────────
   5    │ [100,800-101,119]    │      20           │  31 × 20 = 620  │    640
   6    │ [101,120-101,439]    │      20           │  31 × 20 = 620  │    640
   7    │ [101,440-101,759]    │      20           │  31 × 20 = 620  │    640
   8    │ [101,760-102,079]    │      20           │  31 × 20 = 620  │    640
   9    │ [102,080-102,399]    │      20           │  31 × 20 = 620  │    640
```

**Work ratio:** Q chunk 5-9 (640) vs Q chunk 0 (2) = **320× difference**

Compared to 128k analysis: 512× → The smaller sequence reduces per-chunk work variance.

---

## Work Inventory

```
Light Q chunks (q0-q4): 160 total (32 heads × 5)
  Work per chunk: 2, 4, 6, 8, 10 K chunks
  Total light work: 32 × (2+4+6+8+10) = 32 × 30 = 960 K chunks

Heavy Q chunks (q5-q9): 160 total (32 heads × 5)
  Work per chunk: 640 K chunks each
  Total heavy work: 160 × 640 = 102,400 K chunks

TOTAL WORK: 103,360 K chunks
CORES: 110
IDEAL PER CORE: 103,360 / 110 = 940 K chunks
```

---

## Current vs Balanced Distribution

### Current: Round-Robin Assignment

With 320 Q chunks assigned round-robin across 110 cores:
```
Core 0: (h0,q0), (h0,q1), (h0,q2)  → 2+4+6 = 12 K chunks    ← LIGHT
Core 1: (h0,q3), (h0,q4), (h0,q5)  → 8+10+640 = 658 K chunks ← MIXED
Core 2: (h0,q6), (h0,q7), (h0,q8)  → 640+640+640 = 1,920 K  ← BOTTLENECK
Core 3: (h0,q9), (h1,q0), (h1,q1)  → 640+2+4 = 646 K chunks
...
```

**Current bottleneck: 1,920 K chunks** (cores with 3 heavy chunks)
**Fastest core: ~12 K chunks** (cores with all light chunks)
**Imbalance: 160×**

### Balanced: Zigzag Pairing

Discrete allocation problem:
```
160 heavy chunks ÷ 110 cores = 1.45 per core

Integer solution:
  Let x cores get 1 heavy chunk
  Let y cores get 2 heavy chunks

  x + y = 110
  x + 2y = 160

  Solving: y = 50, x = 60

Result:
  60 cores: 1 heavy chunk = 640 K chunks (from heavy)
  50 cores: 2 heavy chunks = 1,280 K chunks (from heavy) ← BOTTLENECK
```

Light chunk distribution (give to underloaded cores):
```
160 light chunks → 60 underloaded cores
160 / 60 = 2.67 chunks per core

Final distribution:
  60 cores (1 heavy): 640 + ~16 (light) ≈ 656 K chunks
  50 cores (2 heavy): 1,280 K chunks ← BOTTLENECK
```

**Balanced bottleneck: 1,280 K chunks**

---

## Comparison: 100k vs 128k

| Metric | 100k (this analysis) | 128k (reference) |
|--------|---------------------|------------------|
| Total sequence | 102,400 | 131,072 |
| q_chunk_size | 320 | 256 |
| k_chunk_size | 160 | 128 |
| Local Q chunks | 10 | 16 |
| Local K chunks | 20 | 32 |
| Heavy chunks total | 160 | 256 |
| Heavy/core ratio | 1.45 | 2.33 |
| Per-chunk work variance | 320× | 512× |
| Current bottleneck | 1,920 K | 5,120 K |
| Balanced bottleneck | 1,280 K | 3,072 K |
| **Speedup from balancing** | **1.50×** | **1.67×** |
| Balanced utilization | 73.4% | 78.3% |

### Key Insight

The 100k case with larger chunk sizes benefits **LESS** from balancing because:
1. Fewer Q chunks per head (10 vs 16) → fewer heavy chunks overall
2. Heavy/core ratio is lower (1.45 vs 2.33) → smaller discrete allocation gap
3. The "all-heavy" worst case has only 3 chunks vs 5 chunks

---

## Visual Comparison

```
CURRENT (no balancing):
  Fastest cores:  █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ (~12 K, idle 99%)
  Mixed cores:    ██████████████████░░░░░░░░░░░░░░░ (~658 K, idle 66%)
  Slowest cores:  ████████████████████████████████████████ (1,920 K) ← bottleneck

BALANCED (zigzag):
  60 cores:       █████████████████████░░░░░░░░░░░░░ (~656 K, idle 49%)
  50 cores:       ██████████████████████████████████ (1,280 K) ← bottleneck

PERFECT (theoretical):
  All 110 cores:  ████████████████████████░░░░░░░░░░ (940 K each)

Speedup: 1,920 → 1,280 = 1.50×
Remaining gap: 1,280 → 940 = 1.36× (lost to discrete granularity)
```

---

## Summary

| Configuration | L1 Fit? | Work Imbalance | Balanced Speedup | Utilization |
|---------------|---------|----------------|------------------|-------------|
| 100k, q=320, k=160 | NO (2.1×) | 160× | 1.50× | 73.4% |
| 128k, q=256, k=128 | YES | 170× | 1.67× | 78.3% |

### Recommendations

1. **L1 constraint is the blocker**: q=320, k=160 does not fit. Need smaller chunks.

2. **If L1 were not a constraint**: Balanced assignment would give 1.50× speedup, less than the 1.67× from 128k due to fewer heavy chunks.

3. **To maximize balancing benefit**: More Q chunks per head (smaller q_chunk_size) creates more heavy chunks, which spreads better across cores.

4. **Alternative chunk sizes that might fit L1** (rough estimates):
   - q=192, k=96 → ~1,400 KB (might fit)
   - q=160, k=80 → ~1,100 KB (should fit)
   - q=128, k=64 → ~800 KB (definitely fits)
