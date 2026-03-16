# Work Distribution Analysis: ring_joint_sdpa (ring_index=0)

## Executive Summary

The work imbalance in `ring_joint_sdpa` stems from **static round-robin Q chunk assignment** combined with **causal masking's triangular work pattern**. Each Q chunk requires processing a different number of K chunks based on its position, but the current implementation assigns Q chunks evenly to cores without accounting for this variance.

**Key Finding:** The codebase already has a solution pattern (`BALANCED_Q_PARALLEL`) in standard SDPA that pairs early Q chunks (low work) with late Q chunks (high work) on the same core. This pattern is **not applied to ring_joint_sdpa**.

---

## Critical Insight: Inter-Device Data Distribution

### What ring_index=0 Actually Processes

For **causal attention** with `is_balanced=True` in a 32-device ring:

```
                     GLOBAL SEQUENCE (131,072 tokens)
                     64 chunks of 2048 tokens each
┌────────────────────────────────────────────────────────────────────────────┐
│ Chunk 0   │ Chunk 1   │ Chunk 2   │ ... │ Chunk 62  │ Chunk 63           │
│ [0-2047]  │[2048-4095]│[4096-6143]│     │[126k-129k]│ [129024-131071]    │
└────────────────────────────────────────────────────────────────────────────┘

BALANCED DISTRIBUTION (zigzag pairing of early + late chunks):
  Device 0:  chunks [0, 63]  → tokens [0-2047] + [129024-131071]
  Device 1:  chunks [1, 62]  → tokens [2048-4095] + [126976-129023]
  Device 2:  chunks [2, 61]  → tokens [4096-6143] + [124928-126975]
  ...
  Device 15: chunks [15, 48] → tokens [30720-32767] + [98304-100351]
  Device 16: chunks [16, 47] → tokens [32768-34815] + [96256-98303]
  ...
  Device 31: chunks [31, 32] → tokens [63488-65535] + [65536-67583]

For ring_index=0 with is_balanced=True:
  - Local Q tokens: [0-2047] (early) + [129024-131071] (late)
  - Local K/V tokens: same as Q (each device has its own Q,K,V shard)

CAUSAL CONSTRAINT: Q[i] can only attend to K[j] where j ≤ i

  - Q[0-2047] (early): can only attend to K[0-2047] → SMALL work
  - Q[129024-131071] (late): can attend to K[0-131071] → HUGE work
```

**Code Evidence** (`ring_joint_sdpa.cpp:134-139`):
```cpp
const bool ring_iter_does_work = (ring_iter_processes_KV_chunks || (do_joint_kv && L != 0)) &&
                                 !(is_causal && ring_index < ring_id && !is_balanced);

if (!ring_iter_does_work) {
    continue;  // Skip KV from devices where ring_index < ring_id
}
```

For ring_index=0 with `is_balanced=True`:
- The `!is_balanced` term is FALSE, so the skip condition is never triggered
- **All ring iterations process KV** (device 0 sees KV from all 32 devices)

The work asymmetry comes from the Q chunk skip logic, not the ring iteration skip.

### Implication for Our Profile (is_balanced=True)

**Critical Discovery:** The Q chunk skip logic creates asymmetric work:

```cpp
// ring_joint_profile_reader.cpp ~line 200
if (q_chunk < half_sequence && is_balanced && ring_index < ring_id) {
    continue;  // Skip early Q chunks
}
```

For ring_index=0 with is_balanced=True:
- `ring_index < ring_id` → `0 < ring_id`
- **FALSE** for ring_id=0 (local KV) → ALL local Q chunks processed
- **TRUE** for ring_id=1-31 (remote KV) → Only local Q chunks 8-15 processed

**Local Q chunk → Global token mapping for device 0:**
```
Local Q chunks 0-7:  → Global tokens [0-2047]       (chunk 0, early sequence)
Local Q chunks 8-15: → Global tokens [129024-131071] (chunk 63, late sequence)
```

**Resulting Work Distribution:**

```
Local Q │ Global Token    │ ring_iter=0  │ ring_iter=1-31 │ Total K
Chunk   │ Range           │ (local KV)   │ (remote KV)    │ chunks
────────┼─────────────────┼──────────────┼────────────────┼─────────
   0    │ [0-255]         │  2 (causal)  │  SKIPPED       │    2
   1    │ [256-511]       │  4 (causal)  │  SKIPPED       │    4
   ...  │  ...            │  ...         │  SKIPPED       │   ...
   7    │ [1792-2047]     │ 16 (causal)  │  SKIPPED       │   16
────────┼─────────────────┼──────────────┼────────────────┼─────────
   8    │ [129024-129279] │ 32 (all)     │ 31 × 32 = 992  │  1024
   9    │ [129280-129535] │ 32 (all)     │ 31 × 32 = 992  │  1024
   ...  │  ...            │  ...         │  ...           │   ...
  15    │ [130816-131071] │ 32 (all)     │ 31 × 32 = 992  │  1024
```

**Key insight:** Local Q chunks 8-15 map to global positions [129024-131071], which is LATE in the sequence. Due to causality, they can attend to ALL preceding K tokens. But since they're at the end of the sequence, there's no causal restriction - they see everything!

**Work ratio: Q chunk 8-15 (1024 each) vs Q chunk 0 (2) = 512x difference!**

This explains the 116x core-level imbalance:
- Cores with Q chunks 0-7: ~2-16 K chunks total → finish in ~1-20ms
- Cores with Q chunks 8-15: ~1024 K chunks each → take ~100-140ms

---

## Production Configuration

```
Ring: 32 devices, is_balanced=True, is_causal=True
Total sequence: 131,072 tokens (64 global chunks × 2048 tokens)

Device 0 (ring_index=0):
  Global chunks: [0, 63] (zigzag balanced assignment)
  Token ranges: [0-2047] + [129024-131071] = 4096 local tokens

Q shape: [B=1, NH=32, local_seq=4096, d_qk=576]
K/V local shape: [B=1, NH=1, local_seq=4096, d_qk=576]
K/V gathered shape: [B=1, NH=1, total_seq=131072, d_qk=576]

Chunking:
  q_chunk_size = 256 tokens → Sq_chunk_t = 8 tiles
  k_chunk_size = 128 tokens → Sk_chunk_t = 4 tiles

  num_local_q_chunks = 4096 / 256 = 16 Q chunks per head
    - Q chunks 0-7:  global tokens [0-2047]       (early, from chunk 0)
    - Q chunks 8-15: global tokens [129024-131071] (late, from chunk 63)

  num_local_k_chunks = 4096 / 128 = 32 K chunks per device

Parallelization:
  total_q_chunks = B * NH * num_q_chunks = 1 * 32 * 16 = 512 Q chunks
  num_cores = 110
  chunks_per_core ≈ 512 / 110 ≈ 4-5 Q chunks per core
```

---

## Current Work Distribution Algorithm

### Host-Side: Q Chunk Assignment
**File:** `ring_joint_sdpa_program_factory.cpp:741-798`

```cpp
// Pseudo-code of current implementation
total_q_chunks = B * NH * num_q_chunks;  // 512
base_chunks_per_core = total_q_chunks / num_cores;  // 512/110 = 4
extra_chunks = total_q_chunks % num_cores;  // 512%110 = 72

for (core_id = 0; core_id < num_cores; core_id++) {
    chunk_count = base_chunks_per_core + (core_id < extra_chunks ? 1 : 0);

    // Assign consecutive flat chunk indices to this core
    core_work[core_id].global_q_start = next_global_chunk;
    core_work[core_id].global_q_count = chunk_count;
    next_global_chunk += chunk_count;
}

// Decode flat index → (batch, head, q_chunk)
decode_flat_chunk(flat_idx) {
    head_index = flat_idx / num_q_chunks;
    q_chunk = flat_idx % num_q_chunks;
    batch = head_index / NH;
    head = head_index % NH;
    return (batch, head, q_chunk);
}
```

**Key insight:** Chunks are assigned in flat index order: `(b=0,h=0,q=0), (b=0,h=0,q=1), ..., (b=0,h=0,q=15), (b=0,h=1,q=0), ...`

### Reader-Side: Q Chunk Skip for Balanced Mode
**File:** `ring_joint_profile_reader.cpp:200-210`

```cpp
// For each Q chunk assigned to this core
for (global_q_chunk = global_q_start; global_q_chunk < global_q_end; global_q_chunk++) {
    q_chunk = global_q_chunk % num_q_chunks;  // 0-15

    // CRITICAL: Skip early Q chunks for remote KV in balanced mode
    if (q_chunk < half_sequence && is_balanced && ring_index < ring_id) {
        continue;  // Q chunks 0-7 skipped when processing remote KV!
    }

    // Read Q, then iterate over K/V chunks...
    for (k_chunk = 0; k_chunk < iter_num_kv_chunks; k_chunk++) {
        // Reader reads ALL K/V chunks, no causal skip here
        read_block(k_generator, ...);
        read_block(v_generator, ...);
    }
}
```

### Compute-Side: Causal K Chunk Processing
**File:** `compute_common.hpp:1680-1735`

```cpp
// For each Q chunk (after reader's balanced mode skip)
for (q_iter = global_q_start; q_iter < global_q_end; q_iter++) {
    q_chunk = q_iter % num_q_chunks;

    // Compute processes ALL K chunks sent by reader
    // For ring_iter=0 (local KV): causal mask applied
    // For ring_iter>0 (remote KV): no causal restriction

    for (k_chunk = 0; k_chunk < k_chunk_end; k_chunk++) {
        if (ring_iter == 0 && k_chunk >= q_high_idx && is_causal) {
            // Causal skip: drain CB but skip compute
            cb_wait_front(cb_k_in); cb_pop_front(cb_k_in);
            cb_wait_front(cb_v_in); cb_pop_front(cb_v_in);
            continue;
        }
        // QK matmul, softmax, output accumulation...
    }
}
```

**Key insight:** The imbalance is created by the reader's Q chunk skip, not the compute's causal skip.

---

## Work Per Q Chunk (is_balanced=True, ring_index=0)

With balanced mode, Q chunks 0-7 (early global tokens) are skipped for remote KV:

```
Local Q │ Global Token     │ Local KV  │ Remote KV      │ Total K  │ Rel.
Chunk   │ Range            │ (causal)  │ (31 ring_iters)│ Chunks   │
────────┼──────────────────┼───────────┼────────────────┼──────────┼──────
   0    │ [0-255]          │     2     │   SKIPPED      │     2    │ 0.2%
   1    │ [256-511]        │     4     │   SKIPPED      │     4    │ 0.4%
   2    │ [512-767]        │     6     │   SKIPPED      │     6    │ 0.6%
   3    │ [768-1023]       │     8     │   SKIPPED      │     8    │ 0.8%
   4    │ [1024-1279]      │    10     │   SKIPPED      │    10    │ 1.0%
   5    │ [1280-1535]      │    12     │   SKIPPED      │    12    │ 1.2%
   6    │ [1536-1791]      │    14     │   SKIPPED      │    14    │ 1.4%
   7    │ [1792-2047]      │    16     │   SKIPPED      │    16    │ 1.6%
────────┼──────────────────┼───────────┼────────────────┼──────────┼──────
   8    │ [129024-129279]  │    32     │  31 × 32 = 992 │  1024    │ 100%
   9    │ [129280-129535]  │    32     │  31 × 32 = 992 │  1024    │ 100%
  10    │ [129536-129791]  │    32     │  31 × 32 = 992 │  1024    │ 100%
  11    │ [129792-130047]  │    32     │  31 × 32 = 992 │  1024    │ 100%
  12    │ [130048-130303]  │    32     │  31 × 32 = 992 │  1024    │ 100%
  13    │ [130304-130559]  │    32     │  31 × 32 = 992 │  1024    │ 100%
  14    │ [130560-130815]  │    32     │  31 × 32 = 992 │  1024    │ 100%
  15    │ [130816-131071]  │    32     │  31 × 32 = 992 │  1024    │ 100%
```

**Key insight:** Q chunks 8-15 map to global positions [129024-131071]. These are near the END of the 131K sequence, so:
- For local KV: they attend to ALL 32 local K chunks (no causal restriction within local)
- For remote KV: they attend to ALL K from all other devices

**Work ratio:** Q chunk 8-15 (1024 each) vs Q chunk 0 (2) = **512x difference!**

This is the ROOT CAUSE of the 116x core-level imbalance.

---

## Visual: Work Distribution Across Cores

```
                    CURRENT: Round-Robin Assignment (is_balanced=True)
                    ═══════════════════════════════════════════════════

Flat Index:  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16 ...
             │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │
             ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼
Head 0:     [q0][q1][q2][q3][q4][q5][q6][q7][q8][q9][q10][q11][q12][q13][q14][q15]
             ↑ LIGHT work (2-16)  ↑   ↑         HEAVY work (1024 each)         ↑
Head 1:                                                                        [q0]...
...

Core Assignment (72 cores get 5 chunks, 38 cores get 4):
┌─────────┬──────────────────────────────────────────────────────────────────────┐
│ Core 0  │ (h=0,q=0), (h=0,q=1), (h=0,q=2), (h=0,q=3), (h=0,q=4)               │
│         │ K chunks: 2 + 4 + 6 + 8 + 10 = 30         ← ALL LIGHT, fast!       │
├─────────┼──────────────────────────────────────────────────────────────────────┤
│ Core 1  │ (h=0,q=5), (h=0,q=6), (h=0,q=7), (h=0,q=8), (h=0,q=9)               │
│         │ K chunks: 12 + 14 + 16 + 1024 + 1024 = 2090  ← 2 HEAVY chunks!     │
├─────────┼──────────────────────────────────────────────────────────────────────┤
│ Core 2  │ (h=0,q=10), (h=0,q=11), (h=0,q=12), (h=0,q=13), (h=0,q=14)          │
│         │ K chunks: 1024 + 1024 + 1024 + 1024 + 1024 = 5120  ← ALL HEAVY!    │
├─────────┼──────────────────────────────────────────────────────────────────────┤
│ Core 3  │ (h=0,q=15), (h=1,q=0), (h=1,q=1), (h=1,q=2), (h=1,q=3)              │
│         │ K chunks: 1024 + 2 + 4 + 6 + 8 = 1044     ← Mixed                  │
└─────────┴──────────────────────────────────────────────────────────────────────┘

Work Distribution (scale: each █ = 500 K chunks):
  Core 0:     30 K chunks  ░                    ← Finishes in ~1ms
  Core 1:   2090 K chunks  ████░                ← Takes ~30ms
  Core 2:   5120 K chunks  ██████████░          ← BOTTLENECK ~140ms
  Core 3:   1044 K chunks  ██░                  ← Takes ~30ms

Wall-clock = slowest core (all-heavy) ≈ 5120 K chunk units
Imbalance: 5120 / 30 = 170x
```

---

## Root Cause Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│              BALANCED MODE WORK DISTRIBUTION (ring_index=0)               │
│                                                                          │
│  Device 0 has Q tokens from two global chunks:                           │
│    - Chunk 0:  global [0-2047]       → local Q chunks 0-7                │
│    - Chunk 63: global [129024-131071] → local Q chunks 8-15              │
│                                                                          │
│  Q chunks 0-7 (EARLY global positions): ONLY local KV                    │
│  ─────────────────────────────────────────────────────                   │
│  q0 [0-255]:      ██ (2 K chunks)        ← Early in sequence, few K      │
│  q1 [256-511]:    ████ (4)                                               │
│  q2 [512-767]:    ██████ (6)                                             │
│  ...                                                                     │
│  q7 [1792-2047]:  ████████████████ (16)                                  │
│                                                                          │
│  Q chunks 8-15 (LATE global positions): LOCAL + ALL REMOTE KV            │
│  ────────────────────────────────────────────────────────────            │
│  q8  [129024-129279]: ████████████████████████████████ (32 + 992 = 1024) │
│  q9  [129280-129535]: ████████████████████████████████ (1024)            │
│  ...                                                                     │
│  q15 [130816-131071]: ████████████████████████████████ (1024)            │
│                       ↑ Late in sequence = attend to EVERYTHING          │
│                                                                          │
│  Legend: Each █ ≈ 32 K chunks                                            │
└──────────────────────────────────────────────────────────────────────────┘

Work Summary:
  Q chunks 0-7:  Global [0-2047], only local KV → 2-16 K chunks (~1% of max)
  Q chunks 8-15: Global [129k-131k], all KV     → 1024 K chunks each (100%)

  IMBALANCE RATIO: 1024 / 2 = 512x
```

---

## Existing Solution Pattern: BALANCED_Q_PARALLEL

**File:** `sdpa_program_factory.cpp:629-632`

```cpp
uint32_t balanced_q_parallel =
    (is_causal && (q_per_core * q_parallel_factor == q_num_chunks) && (q_per_core % 2 == 0));
if (balanced_q_parallel) {
    defines["BALANCED_Q_PARALLEL"] = "1";
}
```

**File:** `compute_common.hpp:1659-1666` (STANDARD SDPA only)

```cpp
#if defined BALANCED_Q_PARALLEL
    uint32_t q_chunk_div_2 = iter_q_end / 2;
    if (q_iter < q_chunk_div_2) {
        // First half: process forward (low Q chunks)
        q_chunk = local_q_start + q_iter;
    } else {
        // Second half: process backward (high Q chunks)
        uint32_t back_q_iter = q_iter - q_chunk_div_2;
        q_chunk = q_num_chunks - 1 - (local_q_start + back_q_iter);
    }
#else
    q_chunk = local_q_start + q_iter;
#endif
```

### How It Works

Instead of processing Q chunks in order, pair early (low work) with late (high work):

```
WITHOUT BALANCED_Q_PARALLEL (current ring_joint_sdpa, is_balanced=True):
┌─────────┬────────────────────────────────────┬─────────────┐
│ Core    │ Q chunks processed (in order)      │ Total K     │
├─────────┼────────────────────────────────────┼─────────────┤
│ Core A  │ q0, q1, q2, q3, q4                 │2+4+6+8+10=30│
│ Core B  │ q8, q9, q10, q11, q12              │5×1024=5120  │
└─────────┴────────────────────────────────────┴─────────────┘
         Imbalance: 5120/30 = 170x (!)

WITH BALANCED_Q_PARALLEL (hypothetical for ring SDPA):
┌─────────┬────────────────────────────────────┬─────────────┐
│ Core    │ Q chunks processed (balanced)      │ Total K     │
├─────────┼────────────────────────────────────┼─────────────┤
│ Core A  │ q0, q1, q8, q9, q10               │2+4+3×1024=3078│
│ Core B  │ q2, q3, q11, q12, q13             │6+8+3×1024=3086│
└─────────┴────────────────────────────────────┴─────────────┘
         Imbalance: 3086/3078 ≈ 1.0x (balanced!)
```

**Note:** The standard SDPA's BALANCED_Q_PARALLEL assumes gradual work increase (q0→q15).
For ring_joint_sdpa with is_balanced=True, the work is BINARY: q0-7 light, q8-15 heavy.
Simple zigzag pairing won't fully balance - need to pair light chunks with heavy chunks.

---

## Why ring_joint_sdpa Doesn't Use This

The `BALANCED_Q_PARALLEL` pattern is inside `if constexpr (sdpa_type == STANDARD)` block:

```cpp
// compute_common.hpp:1657
if constexpr (sdpa_type == STANDARD) {
    uint32_t q_chunk;
#if defined BALANCED_Q_PARALLEL
    // ... balanced logic ...
#else
    q_chunk = local_q_start + q_iter;
#endif
} else if (sdpa_type == RING) {
    // Ring SDPA uses different decoding (flat index → batch, head, q_chunk)
    const uint32_t nb = q_iter / (NH * q_num_chunks);
    const uint32_t nq = (q_iter % (NH * q_num_chunks)) / q_num_chunks;
    const uint32_t q_chunk = q_iter % q_num_chunks;  // ← Always sequential!
    // ...
}
```

The ring_joint_sdpa uses **flat global Q chunk indexing** (across all batches and heads) rather than per-head local indexing, making the balanced pattern harder to apply directly.

---

## Potential Solutions

### The Real Problem

With `is_balanced=True` and `ring_index=0`, the work split is:
- Q chunks 0-7: Only local KV (2-16 K chunks) → **1% of total work**
- Q chunks 8-15: Local + all remote KV (1010-1024 K chunks) → **99% of total work**

Simple zigzag pairing within Q chunks 0-15 won't help because the work split is binary (0-7 vs 8-15), not gradual.

### Option 1: Separate Core Pools

Assign Q chunks 0-7 to a small pool of cores, Q chunks 8-15 to the rest:

```cpp
// With 110 cores, 32 heads, 16 Q chunks per head = 512 total Q chunks
// Q chunks 0-7: 32 × 8 = 256 chunks, ~1% work each → assign to ~3 cores
// Q chunks 8-15: 32 × 8 = 256 chunks, ~99% work each → assign to ~107 cores

uint32_t light_work_cores = num_cores * (sum_light_work / total_work);  // ~3 cores
uint32_t heavy_work_cores = num_cores - light_work_cores;  // ~107 cores

// Assign Q chunks 0-7 (all heads) to light_work_cores
// Assign Q chunks 8-15 (all heads) to heavy_work_cores
```

**Complexity:** Medium - requires restructuring the assignment loop.

### Option 2: Don't Skip Q Chunks 0-7 for Remote KV

The current skip logic creates the imbalance. Removing the skip would make all Q chunks process all ring iterations:

```cpp
// Current (creates imbalance):
if (q_chunk < half_sequence && is_balanced && ring_index < ring_id) {
    continue;  // Skip Q chunks 0-7 for remote KV
}

// Alternative: Don't skip, let all Q chunks process all KV
// This requires handling the causal mask correctly for remote KV
```

**Complexity:** High - changes the semantics of balanced mode.

### Option 3: Work-Aware Assignment (Host-Side)

Pre-compute work per Q chunk and use greedy bin-packing:

```cpp
// Work per Q chunk for ring_index=0, is_balanced=true:
// Q chunks 0-7: local_k_chunks only (causal)
// Q chunks 8-15: local_k_chunks + 31 * 32 remote K chunks

uint32_t compute_work(uint32_t q_chunk, bool is_balanced, uint32_t ring_size) {
    if (q_chunk < num_q_chunks / 2) {
        // Light work: only local KV
        return (q_chunk + 1) * 2;  // Causal: 2, 4, 6, ..., 16
    } else {
        // Heavy work: local + all remote
        return ((q_chunk + 1) * 2) + (ring_size - 1) * num_local_k_chunks;
    }
}

// Greedy assignment: assign heaviest chunks first to least-loaded cores
```

**Complexity:** Medium - requires computing work estimates at program creation time.

### Option 4: Zigzag Across Work Classes

Pair one light Q chunk (0-7) with multiple heavy Q chunks (8-15):

```cpp
// Each core gets: 1 light chunk + N heavy chunks
// Light chunks: 8 total, ~16 K chunks each
// Heavy chunks: 8 total, ~1000 K chunks each

// With 110 cores and 32 heads:
// - 256 light chunks (32 heads × 8)
// - 256 heavy chunks (32 heads × 8)

// Assign: ~2-3 light + ~2-3 heavy per core to balance
```

**Complexity:** Low-Medium - simple pairing logic.

---

## Theoretical Improvement

### Work Inventory

```
Light Q chunks (q0-q7): 256 total (32 heads × 8)
  Work per chunk: 2, 4, 6, 8, 10, 12, 14, 16 K chunks
  Total light work: 32 × (2+4+6+8+10+12+14+16) = 32 × 72 = 2,304 K chunks

Heavy Q chunks (q8-q15): 256 total (32 heads × 8)
  Work per chunk: 1024 K chunks each
  Total heavy work: 256 × 1024 = 262,144 K chunks

TOTAL WORK: 264,448 K chunks
CORES: 110
IDEAL PER CORE: 264,448 / 110 = 2,404 K chunks
```

### The Discrete Allocation Problem

Heavy chunks are **indivisible units of 1024 K chunks each**.

```
Heavy chunk distribution constraint:
  256 heavy chunks ÷ 110 cores = 2.33 per core

  We must assign integers:
    Let x cores get 2 heavy chunks
    Let y cores get 3 heavy chunks

    x + y = 110
    2x + 3y = 256

    Solving: y = 36, x = 74

  Result:
    74 cores: 2 heavy chunks → 2,048 K chunks (from heavy alone)
    36 cores: 3 heavy chunks → 3,072 K chunks (from heavy alone)
```

### Best Case Distribution with Option 4

```
Strategy: Give all light chunks to the 74 "underloaded" cores

74 cores (2 heavy):
  Heavy: 2 × 1024 = 2,048 K chunks
  Light: 256 ÷ 74 ≈ 3.5 chunks × avg 9 = 31 K chunks
  TOTAL: ~2,079 K chunks

36 cores (3 heavy):
  Heavy: 3 × 1024 = 3,072 K chunks
  Light: 0 (give them nothing extra!)
  TOTAL: 3,072 K chunks  ← BOTTLENECK
```

### Utilization Analysis

```
                        │ Slowest Core │ Ideal Core │ Utilization │
────────────────────────┼──────────────┼────────────┼─────────────┤
Perfect distribution    │    2,404     │   2,404    │    100%     │
Option 4 (best case)    │    3,072     │   2,404    │    78.3%    │
Current (no balancing)  │   ~5,120     │   2,404    │    47.0%    │
```

### Why 78.3% is the Hard Limit

The fundamental constraint is **discrete chunk granularity**:

```
256 heavy chunks ÷ 110 cores = 2.33

Since we can't split chunks:
  - At least 36 cores MUST get 3 heavy chunks
  - These cores will have AT LEAST 3,072 K chunks
  - No amount of light-chunk shuffling can reduce this

Minimum possible max work = 3,072 K chunks
Maximum utilization = 2,404 / 3,072 = 78.3%
```

### Visual Comparison

```
PERFECT (theoretical):
  All 110 cores: ████████████████████████ (2,404 each)
  Wall-clock: 2,404 units

OPTION 4 (best achievable):
  74 cores:     ████████████████████░░░░ (2,079 each)
  36 cores:     ██████████████████████████████░░ (3,072 each) ← bottleneck
  Wall-clock: 3,072 units
  Efficiency: 78.3%

CURRENT (no balancing):
  ~50 cores:    ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ (~600 each, idle 88%)
  ~60 cores:    ██████████████████████████████████████████████░░░░ (~5,120 each)
  Wall-clock: 5,120 units
  Efficiency: 47%
```

### Speedup Summary

| Metric | Current | Option 4 | Perfect |
|--------|---------|----------|---------|
| Slowest core work | 5,120 | 3,072 | 2,404 |
| Utilization | 47% | 78.3% | 100% |
| Relative speedup | 1.0x | **1.67x** | 2.13x |

### Key Insight

**Option 4 captures 78% of the theoretical maximum improvement.** The remaining 22% is lost to the discrete granularity of heavy chunks (1024 K chunks each).

To achieve >78.3% utilization would require either:
1. More cores (so 256/cores < 2.5)
2. Smaller chunk sizes (finer work granularity)
3. Work-stealing at runtime (dynamic load balancing)

---

## Relevant Code Locations

| File | Lines | Description |
|------|-------|-------------|
| `ring_joint_sdpa_program_factory.cpp` | 741-798 | Host-side Q chunk assignment |
| `ring_joint_sdpa.cpp` (kernel) | 73-76, 122-135 | Kernel Q/K iteration setup |
| `compute_common.hpp` | 1659-1666 | BALANCED_Q_PARALLEL pattern |
| `compute_common.hpp` | 1680-1694 | Ring SDPA Q chunk decoding |
| `compute_common.hpp` | 1705-1735 | K chunk iteration with causal skip |
| `sdpa_program_factory.cpp` | 629-632 | BALANCED_Q_PARALLEL enablement |

---

## Recommendations

1. **Short-term (Option 4):** Implement cross-class pairing - assign light Q chunks (0-7) and heavy Q chunks (8-15) together per core.
   - Expected speedup: **1.67x** (from 47% to 78.3% utilization)
   - Hard limit: 78.3% utilization due to discrete chunk granularity
   - Implementation: Modify `ring_joint_sdpa_program_factory.cpp` to interleave light/heavy chunks

2. **Investigation:** Verify the work distribution hypothesis:
   - Instrument cores to log their Q chunk assignments
   - Confirm light-work cores (Q chunks 0-7) finish in <20ms
   - Confirm heavy-work cores (Q chunks 8-15) take >100ms

3. **Beyond 78.3% utilization:** To exceed the Option 4 limit would require:
   - **More cores:** If cores > 128, then 256/cores < 2, no core needs 3 heavy chunks
   - **Finer granularity:** Smaller q_chunk_size or dynamic work splitting
   - **Runtime work-stealing:** Idle cores steal work from busy cores

## Key Insight

The 512x work imbalance is NOT from causal masking alone (which would be ~16x). It's from the **interaction between balanced mode and causal masking**:
- Balanced mode skips Q chunks 0-7 for remote KV
- This means Q chunks 0-7 only process ~1% of the total KV
- Q chunks 8-15 process ~99% of the total KV

This is by design for inter-device load balancing, but creates severe intra-device imbalance.

**The 78.3% utilization ceiling** exists because:
- 256 heavy chunks ÷ 110 cores = 2.33 (non-integer)
- 36 cores must receive 3 heavy chunks (3,072 K chunks each)
- This bottleneck cannot be eliminated by reshuffling light chunks
