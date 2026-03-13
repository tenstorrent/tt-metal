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

## Hypothesis Analysis

### Priority-Ordered Table

| Priority | Hypothesis | Status | Issue | Evidence |
|----------|------------|--------|-------|----------|
| **1** | **H5: DRAM Bandwidth Saturation** | **CRITICAL** | Reader spends 94% of time in DRAM reads (K-READ 59%, V-READ 35%). This is the upstream bottleneck. | K-RESERVE only 8.6% - reader not blocked by compute |
| **2** | **H3: CB Wait Stalls (WAIT-K)** | **CONFIRMED** | Compute waits 16% for K data. Will worsen if compute speeds up. | WAIT-K = 82ms of 515ms total |
| **3** | **H2: QK Matmul Subblock Config** | **SUSPECTED** | QK matmul is 66% of compute - may have suboptimal subblock dims. | Dominates compute but blocked by data |
| **4** | **H4: Work Imbalance Across Cores** | **UNKNOWN** | 512 Q chunks / 110 cores = uneven (4-5 per core). | Need per-core analysis |
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
**Status: UNKNOWN**

**Theory:** With 110 cores and B*NH*num_q_chunks work units, load may be uneven.

**Configuration:**
- B=1, NH=32, num_q_chunks=16
- Total work = 512 Q chunks
- Per core = 512 / 110 = 4.65 chunks (some get 4, some 5)

**Next Steps:** Extract per-core durations from profile_log_device.csv.

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

## Key Insight: Memory-Bound System

**The system is memory-bound, not compute-bound.**

```
Data Flow:
DRAM ──[K-READ]──> Reader CB ──[WAIT-K]──> Compute ──[QK]──> ...
         ↑                         ↑              ↑
    DRAM-limited (94%)      waiting 16%     66% of compute
```

**Consequence:** Optimizing QK matmul alone would shift time from QK to WAIT-K with no net wall-clock improvement. The bottleneck is upstream in DRAM bandwidth.

---

## Optimization Priorities

1. **DRAM Read Optimization** (addresses H5)
   - Increase CB sizes for more buffering/prefetch
   - Ensure reads are coalesced and aligned
   - Consider double-buffering strategies

2. **Reader Prefetching** (addresses H3)
   - Start reading next K/V while compute processes current
   - Overlap DRAM latency with compute

3. **QK Matmul Efficiency** (addresses H2)
   - Only after data bottleneck is addressed
   - Investigate subblock configuration
   - Check for DST register spills

4. **Work Distribution** (addresses H4)
   - Analyze per-core variation
   - Consider load balancing if variance is high

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
