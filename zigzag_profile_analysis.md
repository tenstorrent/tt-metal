# Ring Joint SDPA Profiling Analysis

## Overview

This document captures findings from profiling the `ring_joint_sdpa_profile` operation, a single-device profiling variant of `ring_joint_sdpa` that measures pure compute time without CCL synchronization overhead.

**Experiment Date:** 2026-03-13
**Hardware:** Blackhole (CHIP_FREQ: 1350 MHz, 110 compute cores)
**Configuration:**
- ring_index=0, ring_size=32, total_seq=131072
- batch=1, local_seq=4096
- GQA: nh_q=32, nh_k=1, nh_v=32 (32 Q heads, 1 KV head)
- d_qk=576 (Q/K head dim), d_v=128 (V head dim)
- q_chunk_size=256, k_chunk_size=128

**Experiment Folder:**
`/generated/profiler/reports/2026_03_13_12_52_24/`

**Files:**
| File | Description |
|------|-------------|
| `ops_perf_results_2026_03_13_12_52_24.csv` | High-level op summary |
| `profile_log_device.csv` | Detailed per-zone timestamps (23MB) |
| `tracy_profile_log_host.tracy` | Tracy binary for GUI viewer |

---

## Key Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| DEVICE KERNEL DURATION | 139.4 ms | Total wall-clock time |
| PM IDEAL | 39.8 ms | Performance model ideal |
| FPU UTIL | 28.5% | Low utilization |
| Gap (Actual / Ideal) | 3.5x | Significant optimization headroom |

---

## Profiler Markers Added

### Compute Kernel (TRISC)
| Marker | Location | Purpose |
|--------|----------|---------|
| `WAIT-K` | Before QK matmul | Time waiting for K data from reader |
| `QK` | Q @ K^T matmul | First matmul phase |
| `SOFTMAX` | reduce_max + sub_exp | SFPU operations (max, exp, sum) |
| `OUT` | softmax @ V matmul | Second matmul phase |
| `RESCALE` | Online softmax correction | Rescaling accumulated output |

### Reader Kernel (NCRISC)
| Marker | Location | Purpose |
|--------|----------|---------|
| `K-RESERVE` | cb_reserve_back for K | Wait for compute to consume K |
| `K-READ` | read_block for K | DRAM read time for K chunk |
| `V-RESERVE` | cb_reserve_back for V | Wait for compute to consume V |
| `V-READ` | read_block for V | DRAM read time for V chunk |

### Writer Kernel (BRISC)
| Marker | Location | Purpose |
|--------|----------|---------|
| `WRITER-OUT` | write_output_and_lse | Write output and LSE to DRAM |
| `WRITER-READ-PREV` | read_prev_output_and_lse | Read previous output for accumulation |

---

## Phase Breakdown Results

### Compute Kernel (aggregated across all cores)

| Phase | Calls | Total Cycles | Time (ms) | % of Compute |
|-------|-------|--------------|-----------|--------------|
| WAIT-K | 8,418 | 111.5M | 82.6 | 16.0% |
| QK | 8,334 | 386.5M | 286.3 | 55.6% |
| SOFTMAX | 8,211 | 39.1M | 28.9 | 5.6% |
| OUT | 8,154 | 116.7M | 86.5 | 16.8% |
| RESCALE | 7,350 | 41.7M | 30.9 | 6.0% |
| **TOTAL** | - | 695.4M | 515.1 | 100% |

### Compute Breakdown (excluding WAIT-K)

| Phase | % of Actual Compute |
|-------|---------------------|
| **QK** | **66.2%** |
| OUT | 20.0% |
| RESCALE | 7.1% |
| SOFTMAX | 6.7% |

### Reader Kernel

| Phase | Calls | Total Cycles | Time (ms) | % of Reader |
|-------|-------|--------------|-----------|-------------|
| K-RESERVE | 3,520 | 12.8M | 9.5 | 5.6% |
| K-READ | 3,410 | 136.7M | 101.3 | 59.1% |
| V-RESERVE | 3,410 | 1.5M | 1.1 | 0.7% |
| V-READ | 3,410 | 80.2M | 59.4 | 34.7% |
| **TOTAL** | - | 231.2M | 171.3 | 100% |

---

## Work Distribution Across Cores

### Table 1: Kernel Duration (ms) - Logical Coordinates

```
 y\x|    0     1     2     3     4     5     6     7     8     9    10
----+------------------------------------------------------------------
   0|   6.7  59.4 138.4  32.8  33.0 138.3  58.9   6.5 137.5  84.5   5.6
   1| 112.9 112.7   7.4  86.6 139.4   7.2  59.8 138.5  32.7  32.3 136.8
   2|  58.2   6.1 137.5  84.7   6.0 111.2 111.0   5.6  84.1 136.5   4.5
   3|  57.3 136.5  30.6  31.0 136.3  56.9   4.7 135.9  82.9   4.0 109.1
   4| 109.5   4.0  83.2 136.2   3.8  56.8 135.9  29.8  30.1 135.2  55.5
   5|   3.4 135.1  82.2   3.3 108.7 108.8   3.0  82.1 135.1   2.7  55.6
   6| 134.8  28.5  29.1 134.6  55.1   2.6 107.5 108.1   1.8   2.1 107.4
   7| 108.0   1.4   2.0 107.4 108.0   1.4   2.0 107.4 108.0   1.4   2.0
   8| 107.3 108.0   1.3   1.9 107.3 107.9   1.3   1.9 107.3 107.9   1.3
   9|   1.8 107.3 107.9   1.2   1.8 107.3 107.9   1.2   1.8 107.3 107.9
```

### Table 2: Visual Pattern - Symbol Indicators

```
Legend:  .=<20ms   o=20-60ms   *=60-100ms   #=>100ms

 y\x|  0  1  2  3  4  5  6  7  8  9 10
----+---------------------------------
   0|  .  o  #  o  o  #  o  .  #  *  .
   1|  #  #  .  *  #  .  o  #  o  o  #
   2|  o  .  #  *  .  #  #  .  *  #  .
   3|  o  #  o  o  #  o  .  #  *  .  #
   4|  #  .  *  #  .  o  #  o  o  #  o
   5|  .  #  *  .  #  #  .  *  #  .  o
   6|  #  o  o  #  o  .  #  #  .  .  #
   7|  #  .  .  #  #  .  .  #  #  .  .
   8|  #  #  .  .  #  #  .  .  #  #  .
   9|  .  #  #  .  .  #  #  .  .  #  #
```

### Work Distribution Summary

| Symbol | Duration | Core Count | Observation |
|--------|----------|------------|-------------|
| `.` | <20ms | 36 | Idle 85%+ of kernel time |
| `o` | 20-60ms | 20 | Medium utilization |
| `*` | 60-100ms | 8 | High utilization |
| `#` | >100ms | 46 | Gate wall-clock time |

### Key Observations

1. **Checkerboard Pattern**: Fast (`.`) and slow (`#`) cores alternate regularly across the grid - this is NOT random but reflects the Q chunk assignment pattern.

2. **Round-Robin Assignment**: Q chunks are assigned round-robin across cores, spreading causal work imbalance across the grid rather than clustering it.

3. **Causal Work Variance**:
   - Q chunk 0 → processes 3 K chunks → ~1-7ms
   - Q chunk 15 → processes 32 K chunks → ~139ms
   - 10x work difference per Q chunk

4. **Wall-Clock Impact**:
   - Kernel duration: 139ms (slowest core at x=4, y=1)
   - Mean duration: 65ms
   - 36 cores finish in <20ms and sit idle for remaining ~120ms

5. **Potential Speedup**: ~2.1x improvement possible (139ms → 65ms) from work rebalancing alone.

---

## Hypothesis Analysis

### Priority-Ordered Table

| Priority | Hypothesis | Status | Issue | Evidence |
|----------|------------|--------|-------|----------|
| **1** | **H4: Work Imbalance Across Cores** | **CRITICAL** | 116x imbalance: fastest core 1.2ms, slowest 139ms. Wall-clock gated by slowest. | 36 cores <20ms, 46 cores >100ms |
| **2** | **H5: DRAM Bandwidth Saturation** | **CRITICAL** | Reader spends 94% of time in DRAM reads (K-READ 59%, V-READ 35%). | K-RESERVE only 8.6% - reader not blocked by compute |
| **3** | **H3: CB Wait Stalls (WAIT-K)** | **CONFIRMED** | Compute waits 16% for K data. Will worsen if compute speeds up. | WAIT-K = 82ms of 515ms total |
| **4** | **H2: QK Matmul Subblock Config** | **SUSPECTED** | QK matmul is 66% of compute - may have suboptimal subblock dims. | Dominates compute but blocked by data |
| **5** | **H6: Causal Masking Overhead** | **CONFIRMED** | ring_index=0 does full causal triangle; ring_index=31 is 14% faster. | Expected, not actionable |
| **6** | **H1: SFPU Dominates** | **REJECTED** | SOFTMAX (SFPU ops) only 6.7% of compute. | Not a bottleneck |

### Detailed Hypothesis Descriptions

#### H1: SFPU Operations Dominate Non-FPU Time
**Status: REJECTED**

**Theory:** FPU utilization only measures tensor FPU ops (matmul). SDPA has heavy SFPU ops (exp, max, sum, recip) that use different hardware and aren't counted in FPU util.

**Finding:** SOFTMAX phase (which contains all SFPU ops) is only 6.7% of compute time. The low FPU utilization is NOT due to SFPU dominance.

#### H2: Matmul Subblock Inefficiency
**Status: SUSPECTED**

**Theory:** With q_chunk=256 (8 tiles), k_chunk=128 (4 tiles), the matmul subblock config may not be optimal for DST register size.

**Finding:** QK matmul consumes 66% of actual compute time. This is the dominant phase, but optimizing it would just shift time to WAIT-K unless data delivery is also improved.

**Next Steps:** Log subblock dimensions from `determine_largest_subblock_size()`.

#### H3: CB Wait Stalls in Compute
**Status: CONFIRMED**

**Theory:** Circular buffer waits (`cb_wait_front`) may introduce stalls if reader cannot keep up with compute.

**Finding:** WAIT-K is 16% of compute time. Compute is data-bound - it finishes processing faster than reader can deliver new K chunks.

**Implication:** Any compute optimization will increase WAIT-K unless reader throughput improves.

#### H4: Work Imbalance Across Cores
**Status: CRITICAL**

**Theory:** With 110 cores and B*NH*num_q_chunks work units, load may be uneven.

**Finding:** Massive imbalance confirmed:
```
Cores: 110  |  Min: 1.19ms  |  Max: 139.39ms  |  Imbalance: 116.9x

Duration Histogram:
  [  0- 20) ms: 36 cores  (idle most of the time)
  [ 20- 40) ms: 10 cores
  [ 40- 60) ms: 10 cores
  [ 60- 80) ms:  0 cores
  [ 80-100) ms:  8 cores
  [100-120) ms: 28 cores
  [120-140) ms: 18 cores  (bottleneck cores)
```

**Root Cause:** Causal masking creates triangular work pattern:
- Q chunk 0 processes only 3 K chunks
- Q chunk 15 processes all 32 K chunks
- Cores assigned to early Q chunks finish 10-100x faster

**Impact:** Wall-clock time is gated by the slowest core (139ms), while 36 cores sit idle after ~20ms.

**Potential Solutions:**
1. Work-stealing between cores
2. Dynamic Q chunk assignment
3. Rebalance Q chunks to equalize K chunk counts

#### H5: DRAM Bandwidth Saturation
**Status: CRITICAL**

**Theory:** Reader kernel may be hitting DRAM bandwidth limits when reading K/V chunks.

**Finding:** Reader spends 94% of time in actual DRAM reads:
- K-READ: 59.1%
- V-READ: 34.7%
- K-RESERVE: 5.6% (waiting for compute)
- V-RESERVE: 0.7%

The reader is NOT waiting on compute (K-RESERVE is small). It's limited by DRAM read speed.

**Next Steps:**
1. Calculate theoretical DRAM bandwidth requirement
2. Compare against Blackhole DRAM bandwidth specs
3. Check if reads are coalesced and aligned
4. Investigate prefetching opportunities

#### H6: Causal Masking Overhead for ring_index=0
**Status: CONFIRMED (Expected)**

**Theory:** ring_index=0 processes all K chunks for all Q chunks (full causal triangle), while ring_index=31 skips many due to causality.

**Evidence:**
- ring_index=0: 139ms, 28.6% FPU util
- ring_index=31: 122ms, 32.5% FPU util (14% faster)

This is expected behavior from the causal attention pattern, not a bug.

---

## Key Insights

### 1. Work Imbalance is the Primary Bottleneck

**116x imbalance between cores due to causal masking.**

```
Core Duration Distribution:
  Fastest cores: ~1.2ms   (36 cores idle after <20ms)
  Slowest cores: ~139ms   (18 cores doing most work)

Wall-clock time = slowest core = 139ms
Theoretical if balanced = ~65ms (mean)
```

**Consequence:** Even with perfect compute/memory optimization, wall-clock is gated by the slowest cores processing the largest Q chunks.

### 2. System is Also Memory-Bound

**The slowest cores are memory-bound, not compute-bound.**

```
Data Flow (on busy cores):
DRAM ──[K-READ]──> Reader CB ──[WAIT-K]──> Compute ──[QK]──> ...
         ↑                         ↑              ↑
    DRAM-limited (94%)      waiting 16%     66% of compute
```

**Consequence:** Optimizing QK matmul alone would shift time from QK to WAIT-K with no net improvement.

---

## Optimization Priorities

1. **Work Rebalancing** (addresses H4) - **TOP PRIORITY**
   - Rebalance Q chunk assignment to equalize K chunk counts
   - Consider work-stealing between cores
   - Dynamic scheduling instead of static assignment

2. **DRAM Read Optimization** (addresses H5)
   - Increase CB sizes for more buffering/prefetch
   - Ensure reads are coalesced and aligned
   - Consider double-buffering strategies

3. **Reader Prefetching** (addresses H3)
   - Start reading next K/V while compute processes current
   - Overlap DRAM latency with compute

4. **QK Matmul Efficiency** (addresses H2)
   - Only after work imbalance and data bottlenecks are addressed
   - Investigate subblock configuration
   - Check for DST register spills

---

## Files Modified for Profiling

| File | Changes |
|------|---------|
| `kernels/compute/compute_common.hpp` | Added WAIT-K, QK, SOFTMAX, OUT, RESCALE markers |
| `kernels/dataflow/ring_joint_profile_reader.cpp` | Split K/V into RESERVE + READ markers |

---

## Reproduction Commands

```bash
# Run profiled test
unset TT_METAL_WATCHER
python -m tracy -p -r -v -m pytest \
    "tests/ttnn/unit_tests/operations/sdpa/test_ring_joint_sdpa_profile.py::test_ring_joint_sdpa_profile_production_scale[ring_index=0]" \
    -v -s

# Analyze results
cat generated/profiler/reports/<timestamp>/ops_perf_results_*.csv
grep -E "QK|SOFTMAX|OUT|RESCALE|WAIT-K" generated/profiler/reports/<timestamp>/profile_log_device.csv | head -100
```
