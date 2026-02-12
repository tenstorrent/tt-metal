# Strategies to Increase ND Failure Rate

The fetch-queue timeout is rare: 11+ stress-test runs with no failure. These strategies aim to amplify the stall/dispatch race to reach ~30% failure rate for analysis.

---

## Root Cause Recap

- **Stall point**: Each `copy_host_to_device_tensor` triggers `add_dispatch_wait_with_prefetch_stall`.
- **Race**: Prefetcher (CQ1) hits `CQ_PREFETCH_CMD_STALL` and waits for dispatch to signal. If dispatch is slow or stuck, prefetcher hangs.
- **Overlap**: CQ0 runs trace; CQ1 does copies. Both CQs active = more contention.

---

## Strategy 1: More Stall Points Per Unit Time (Recommended)

**Idea**: Increase the number of `copy_host_to_device_tensor` calls (each = 1 stall).

| Approach | How | Effect |
|----------|-----|--------|
| **Multiple copies per iteration** | Do N copies to the same buffer per iteration (copy, wait, copy, wait, ...) | N× stalls per iteration |
| **More iterations** | 5000–10000 instead of 1100 | ~5–10× more stalls per run |
| **Combined** | 5 copies × 2000 iters = 10,000 stalls per run | ~10× more than original |

**Implementation**: See `vit_n300/test_vit_2cq_copy_stress.py` and run with `./vit_n300/stress_test_copy_stress.sh`

---

## Strategy 2: Minimal CQ0 Work (Faster Iterations)

**Idea**: Shorten each iteration so more iterations fit in the same time.

| Approach | How | Effect |
|----------|-----|--------|
| **Smaller batch** | `batch_size=1` instead of 8 | ~fewer ops per trace |
| **Trivial trace** | Replace ViT with tiny op (e.g., single reshape/add) | Much faster CQ0; more overlap with copies |
| **Shorter trace** | Use a smaller model or fewer layers | Less CQ0 work per iteration |

**Trade-off**: Trivial trace needs a different test setup. Smaller batch is easy.

---

## Strategy 3: Tighter Overlap / More Contention

**Idea**: Maximize simultaneous CQ0 and CQ1 activity.

| Approach | How | Effect |
|----------|-----|--------|
| **Reduce host-side waits** | Fewer `wait_for_event` calls so more commands are enqueued before waiting | More commands in flight |
| **Multiple buffers** | Copy to N different buffers so copies can overlap | More CQ1 traffic |
| **Pipeline depth** | Enqueue several copies before first wait | More prefetcher pressure |

---

## Strategy 4: Sharded Buffer Stress

**Idea**: The stall path is used for sharded buffers (`ShardedBufferWriteDispatchParams`). Ensure we hit it.

- Use sharded DRAM config (already the case for ViT).
- Use multiple shards / smaller pages to trigger more per-copy work.

---

## Strategy 5: Parameter Tuning (Quick Wins)

| Param | Current | Try | Rationale |
|-------|---------|-----|-----------|
| `num_warmup_iterations` | 100 | 500 | More warmup before measurement |
| `num_measurement_iterations` | 1000 | 5000 | More stall points |
| `copies_per_iteration` | 1 | 5–10 | Multiple stalls per iteration |
| `batch_size` | 8 | 1 or 4 | Faster iterations, different timing |

---

## Target: ~30% Failure Rate

1. Run the modified stress test with 5–10 copies/iter and 2000–5000 iterations.
2. If still no failure, increase copies per iteration to 20–50.
3. If needed, add a trivial-trace variant that replaces ViT with a single tiny op.
