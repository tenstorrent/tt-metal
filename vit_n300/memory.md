# ViT N300 Non-Deterministic Hang Investigation -- Comprehensive Memory

## Problem Statement

The `vit-N300-func` CI test non-deterministically hangs with:

```
RuntimeError: TT_THROW @ .../system_memory_manager.cpp:572: tt::exception
TIMEOUT: device timeout in fetch queue wait, potential hang detected
```

The hang is a **Tensix matmul deadlock** in `bmm_large_block_zm_fused_bias_activation` using block-sharded L1 + `MatmulMultiCoreReuseMultiCastProgramConfig` on an 8x8 grid. It was observed twice in CI (ND_failure1.log, ND_failure2.log).

---

## Root Cause (from CI triage)

**Deadlock at physical core (7,3) = virtual core 9-5 on device 0:**

| RISC | Kernel | Stuck At | Meaning |
|------|--------|----------|---------|
| BRISC (kernel 66) | reader_bmm_tile_layout_in1_receiver_writer_padding | cb_wait_front() line 225 | Waiting for output CB tiles |
| TRISC0 (kernel 67) | bmm_large_block_zm_fused_bias_activation | wait_for_next_context() in add_tiles_bcast_rows line 430 | Unpacker waiting for next context |
| TRISC1 (kernel 67) | bmm_large_block_zm_fused_bias_activation | set_dst_write_addr() in add_tiles_bcast_rows line 430 | Math engine executing |
| TRISC2 (kernel 67) | bmm_large_block_zm_fused_bias_activation | llk_push_to_brisc() / cb_push_back | Packer trying to push output tile |

**Deadlock chain:** TRISC2 (packer) cannot push output tiles because BRISC (writer) hasn't consumed previous ones. BRISC is waiting for output tiles but can't see them. This is a **circular buffer semaphore race condition** in the FUSE_BIAS path of the matmul compute kernel.

---

## CI Test Environment (where hang occurs)

- **Test:** `test_demo_vit_ttnn_inference_perf_e2e_2cq_trace.py::test_vit`
- **Hardware:** N300 (2 Wormhole chips via MeshDevice)
- **Config:** `l1_small_size=32768, num_command_queues=2, trace_region_size=1753088`
- **Batch size:** 8
- **CI env variables:** `TT_METAL_OPERATION_TIMEOUT_SECONDS=5`
- **Pattern:** 2CQ trace replay with:
  - CQ0: reshard, execute_trace (full ViT model: ~60 matmul ops per trace), to_memory_config (output to DRAM)
  - CQ1 (concurrent): copy_host_to_device, from_device (output to host)
  - Synchronized via record_event / wait_for_event between CQs
- **Iterations:** 100 warmup + 1000 measurement = **1100 trace replays** = ~66K matmul ops
- **Normal throughput:** ~1323 samples/sec (N300), ~1470 (N150)

---

## Experiments Performed (This Session)

### 1. Running the Actual CI Test

| Run | Config | Samples/sec | Duration | Hang? | Total Matmul Ops |
|-----|--------|------------|----------|-------|-----------------|
| 1 | Watcher only, timeout=10s | 261 | 70s | No | ~66K |
| 2 | Watcher + DEBUG_DELAY=2000 (not properly configured) | 259 | 71s | No | ~66K |
| 3-7 (loop of 5) | No watcher, timeout=5s | ~1198 | ~11.5s each | No | ~330K |
| 8-12 (loop of 5) | Watcher + DEBUG_DELAY=100000 (NOT properly configured) | ~260 | ~38-71s each | No | ~330K |
| 13 | Watcher + DEBUG_DELAY=5000, one core only (format error) | 199 | 82s | No | ~66K |
| 14 | Watcher + DEBUG_DELAY=5000, ALL 128 cores properly configured | 193 | 53s | No | ~66K |

**Total ops across all runs: ~990K matmul operations, zero hangs.**

### 2. Previous Session's Stress Tests (isolated matmul, not full ViT)

- test_matmul_with_bias: 10000 iters x 4 configs = 40K ops per run, ~8000 ops/s
- test_matmul_deadlock_stress_2cq: 10000 iters x 4 configs + 5 CQ1 copies = 40K matmul + 50K copy ops
- Combined previous session total: ~320K+ matmul operations, zero hangs

---

## CRITICAL DISCOVERY: TT_METAL_WATCHER_DEBUG_DELAY Configuration

**The debug delay feature was NOT working in most experiments!**

Setting `TT_METAL_WATCHER_DEBUG_DELAY=<cycles>` alone is NOT sufficient. You MUST also specify which cores to delay using:

```bash
# Environment variables for debug delay:
export TT_METAL_WATCHER=1                                    # Required: enable watcher
export TT_METAL_WATCHER_DEBUG_DELAY=5000                     # Delay cycles per NoC transaction
export TT_METAL_WRITE_DEBUG_DELAY_CORES='(0,0)-(7,7)'       # Target cores for write delays
export TT_METAL_ATOMIC_DEBUG_DELAY_CORES='(0,0)-(7,7)'      # Target cores for atomic delays
# Optional:
export TT_METAL_READ_DEBUG_DELAY_CORES='(0,0)-(7,7)'        # Target cores for read delays
```

**Core range format:**
- Single core: `0,0` or `3,4`
- Range: `(0,0)-(7,7)` (parentheses required!)
- `all`, `worker`, `dispatch` -- DO NOT WORK for delay feature (only populate all_cores flag, not cores map; the watcher server delay init iterates cores only)

**Verification:** Correct configuration produces 128 "Configured Watcher debug delays" log messages (64 per device on N300). Without proper core targeting, you get 0 or 2 messages and the delay has zero effect on performance.

**Performance impact with proper configuration:**
- No watcher: ~1198 samples/sec
- Watcher only: ~260 samples/sec (4.6x slower due to NoC sanitization)
- Watcher + 5K delay on all cores: ~193 samples/sec (6.2x slower total)

---

## Device Profiler (NoC Events)

**How to use:**
```bash
export TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1
export TT_METAL_DEVICE_PROFILER_NOC_EVENTS_RPT_PATH=/path/to/reports
# NOTE: Conflicts with TT_METAL_WATCHER -- cannot use both simultaneously
# Requires Tracy-enabled build
```

**Output:** JSON files in the report path: `noc_trace_dev{N}_ID{RuntimeID}[_traceID{N}].json`

**Findings:**
- Captures dispatch-level NoC operations (BRISC + NCRISC reads/writes)
- **Does NOT capture kernel-internal multicast operations** (the ones involved in the deadlock)
- Event types seen: READ, READ_SET_STATE, READ_WITH_STATE, WRITE_, READ/WRITE_BARRIER_START/END
- No MULTICAST events in trace replay data
- Max timing gap in a passing run: 220 cycles (tiny -- no anomalies)

**Conclusion:** The profiler is NOT useful for detecting "near miss" patterns in the matmul multicast deadlock because it doesn't instrument kernel-internal NoC transactions.

---

## Why the Hang Cannot Be Reproduced Easily

1. **Extremely rare:** Over 1.3M matmul operations executed across all experiments with zero hangs
2. **Full model complexity required:** The CI test runs the complete ViT model (~60 ops per trace including layernorm, attention, softmax, etc.), not just isolated matmuls. The memory allocation pattern and operation ordering may be critical.
3. **N300 dual-chip specifics:** MeshDevice with 2 chips introduces ethernet dispatch overhead (commented in CI test: "there's a problem with eth dispatch, hence lower perf")
4. **Concurrent 2CQ traffic pattern:** The CI test has a specific interleaving of CQ0 (compute trace) and CQ1 (host-device copies) synchronized by events -- more complex than our stress tests
5. **Thermal/environmental:** CI runs in Docker containers with potentially different thermal/timing characteristics
6. **Memory fragmentation:** The full model allocates and deallocates many buffers; a specific allocation pattern might be needed

---

## Hardware Details

- **Board:** N300 (2 Wormhole B0 chips, PCIe 0000:04:00.0)
- **Device 0 harvesting:** tensix=0x220 (rows 5,9 harvested) -> 56 L1 banks
- **Device 1 harvesting:** tensix=0x300 (rows 8,9 harvested) -> 56 L1 banks
- **KMD:** 2.6.0, UMD firmware bundle 19.4.2
- **Dispatch:** ETH dispatch (DispatchCoreType.ETH, DispatchCoreAxis.ROW)

---

## Debug Tools Summary

| Tool | Usefulness for This Bug | Notes |
|------|------------------------|-------|
| TT_METAL_OPERATION_TIMEOUT_SECONDS | **Essential** -- detects hangs | Set to 5s (CI default). Resets on any dispatch progress. |
| TT_METAL_WATCHER | **Good for post-hang diagnosis** | Captures core state snapshot. 1s polling too slow for "near miss". |
| TT_METAL_WATCHER_DEBUG_DELAY | **Theoretically useful** -- widens race windows | Must configure target cores properly (see above). 5K delay = 26% slowdown. Not yet tested with large enough delay on all cores in a long loop. |
| DPRINT | **Not useful** -- too much overhead | Crashes machine with high volume. Event-driven logging is "pure noise" (fires on expected waits). Overhead prevents OPERATION_TIMEOUT from triggering. |
| Device Profiler (NoC events) | **Not useful for this bug** | Only captures dispatch-level NoC, not kernel-internal multicast. Conflicts with Watcher. |
| Watcher Ring Buffer | **Untested but promising** | 32-entry per-core circular buffer. Low overhead. Could log CB semaphore states. Requires kernel code changes. |

---

## Recommended Next Steps

### Highest Priority: Reproduce the hang

1. **Run the actual CI test in a long loop** with properly configured debug delay:
   ```bash
   export TT_METAL_WATCHER=1
   export TT_METAL_WATCHER_DEBUG_DELAY=10000
   export TT_METAL_WRITE_DEBUG_DELAY_CORES='(0,0)-(7,7)'
   export TT_METAL_ATOMIC_DEBUG_DELAY_CORES='(0,0)-(7,7)'
   export TT_METAL_OPERATION_TIMEOUT_SECONDS=15
   # Run: ./vit_n300/scripts/loop_vit_ci_test.sh 100 10000
   ```
   This needs ~100+ runs with proper delay to have a chance. Previous experiments used incorrect delay config.

2. **Instrument matmul kernels with Watcher Ring Buffer** (WATCHER_RING_BUFFER_PUSH) to log CB semaphore states at critical points in:
   - bmm_large_block_zm_fused_bias_activation.cpp (compute kernel, TRISC)
   - reader_bmm_tile_layout_in1_receiver_writer_padding.cpp (BRISC writer)
   - reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp (NCRISC reader)

   This is low-overhead and data persists in L1 -- readable after a hang via watcher dump.

3. **Try running with MCAST_INPUT_BUFFERING_DEPTH=1** combined with the properly configured debug delay (this combination was never tested -- previous MCAST_INPUT_BUFFERING_DEPTH=1 tests had no delay).

### Alternative: Analyze the CI deadlock more deeply

4. **Examine the exact add_tiles_bcast_rows call at line 430** of bmm_large_block_zm_fused_bias_activation.cpp -- both TRISC0 (unpack) and TRISC1 (math) were stuck there. This is the FUSE_BIAS path where bias tiles are broadcast-added to output tiles. The unpacker-math-packer pipeline might have a specific ordering issue in this bias path.

5. **Check if the issue is in llk_push_to_brisc** -- TRISC2 was stuck trying to signal BRISC that output tiles are ready. If this semaphore increment doesn't reach BRISC (or BRISC reads a stale value), the deadlock occurs.

---

## Files Modified/Created

- `/tt-metal/vit_n300/tests/test_matmul_deadlock_stress.py` -- Stress test with multiple matmul configs, 2CQ, bias
- `/tt-metal/vit_n300/scripts/stress_test_matmul.sh` -- Runner script
- `/tt-metal/vit_n300/scripts/loop_vit_ci_test.sh` -- Loop script for actual CI test
- `/tt-metal/vit_n300/INVESTIGATION_STATUS.md` -- Previous investigation status
- `/tt-metal/vit_n300/README.md` -- Usage instructions
- `/tt-metal/vit_n300/logs/` -- All experiment logs
- `/tt-metal/vit_n300/logs/noc_profiler/` -- Device profiler NoC trace data

---

## Key Insight

The most important finding from this session is that **all previous TT_METAL_WATCHER_DEBUG_DELAY experiments were ineffective** because the target cores were not properly specified. The delay feature requires explicit core range configuration via `TT_METAL_WRITE_DEBUG_DELAY_CORES='(0,0)-(7,7)'` (with parentheses). The `all` keyword does not work for the delay feature. Future experiments should use the correct configuration to actually amplify race windows.
