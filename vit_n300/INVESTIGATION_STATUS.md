# ViT N300 Non-Deterministic Device Hang — Investigation Status

**Last updated**: 2026-02-12
**Status**: Root cause identified, reproduction blocked by wrong code path in stress test

---

## Problem Statement

The `vit-N300-func` CI test non-deterministically hangs with a device timeout in
`fetch_queue_reserve_back`. Two CI failure logs have been captured:
- `ND_failure.log` — first occurrence
- `ND_failure2.log` — second occurrence, confirming the pattern

Both show the same causal chain: a **Tensix matmul deadlock** inside
`bmm_large_block_zm_fused_bias_activation` on an 8x8 block-sharded multicast grid.

---

## Root Cause (High Confidence)

The deadlock is a **circular dependency between BRISC (output writer) and TRISC2 (packer)**
on the output circular buffer, occurring specifically in the **non-sharded output path with
fused bias+activation**.

### Stuck state from CI failure logs (both failures identical):

| RISC | Stuck at | Role |
|------|----------|------|
| **brisc** | `cb_wait_front` | Waiting for pack to produce output tiles |
| **trisc2** | `llk_push_to_brisc` / `cb_push_back` | Packed tiles, waiting for brisc to consume (free CB space) |
| **trisc1** | `add_tiles_bcast_rows` | Stuck in bias+activation compute |
| **trisc0** | `wait_for_next_context` | Waiting for context switch |
| **ncrisc** | `wait_for_brisc_notification` | Waiting for brisc to signal |

Additional stuck components:
- CQ0 dispatch core: `process_go_signal_mcast_cmd()` — can't send next go signal
- CQ1 prefetcher: `process_stall()` — cascading stall → host-side `fetch_queue_reserve_back` timeout

### Why it deadlocks

In the **non-sharded output** path, BRISC alternates between:
1. Reading in1 via multicast (`cb_reserve_back` → `noc_semaphore_wait` → `cb_push_back`)
2. Writing output tiles to DRAM (`cb_wait_front` → `noc_async_write` → `cb_pop_front`)

The output CB is shared between pack (producer) and writer (consumer). If timing aligns
such that:
- Pack fills the output CB and blocks on `cb_push_back` (CB full)
- Writer hasn't consumed any tiles yet and blocks on `cb_wait_front` (CB empty from writer's perspective due to metadata race)

...you get a permanent circular wait. The bias multicast path adds additional synchronization
that can delay the pipeline enough to trigger this race.

### Why it's non-deterministic

The race requires precise alignment of:
- Multicast arrival timing (in0 and in1)
- Compute pipeline throughput
- NoC traffic from CQ1 data movement
- The cold-start phase of each matmul (bw=0), where 100% contention exists

---

## What We Built

### Files created/modified

```
vit_n300/
├── INVESTIGATION_STATUS.md          ← this file
├── README.md                        ← usage instructions
├── ND_failure.log                   ← first CI failure log
├── ND_failure2.log                  ← second CI failure log
├── scripts/
│   ├── stress_test_matmul.sh        ← matmul stress test runner (with DPRINT support)
│   ├── run_vit_n300.sh              ← original ViT model runner
│   ├── stress_test_vit_n300.sh      ← ViT model stress loop
│   └── stress_test_copy_stress.sh   ← CQ1 copy stress
├── tests/
│   ├── test_matmul_deadlock_stress.py  ← isolated matmul stress test
│   └── test_vit_2cq_copy_stress.py    ← CQ1 copy contention test
├── explanations/
│   └── STRESS_STRATEGY.md           ← detailed deadlock analysis
└── logs/                            ← test output logs (gitignored)
```

### Kernel modifications (still in tree)

1. **`ttnn/.../matmul_utilities.hpp`**:
   `MCAST_INPUT_BUFFERING_DEPTH` changed from `2` to `1` (removes double-buffering
   to increase back-pressure). **Revert before merging to main.**

2. **`ttnn/.../reader_bmm_tile_layout_in1_receiver_writer_padding.cpp`**:
   Event-driven DPRINT: prints `BR:in1_WAIT` only when in1 semaphore is INVALID (contention).

3. **`ttnn/.../reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp`**:
   Event-driven DPRINT: prints `NC:SEND_WAIT` when receivers haven't acknowledged,
   `NC:in0_WAIT` when in0 data not ready.

4. **`ttnn/.../compute/bmm_large_block_zm_fused_bias_activation.cpp`**:
   All DPRINT removed (static counters reset per trace replay, making periodic sampling
   useless). The `#include "api/debug/dprint.h"` was also removed.

### Stress test (`test_matmul_deadlock_stress.py`)

- 4 ViT matmul configs (QKV, self_output, FF1, FF2) on 8x8 grid
- 10,000 iterations × 4 matmuls = 40,000 matmul ops per run
- 2CQ variant adds 5 CQ1 host↔device copies per iteration (50,000 total)
- Wide matmul variants (6144, 4096 columns) for extra back-pressure
- Uses `L1_BLOCK_SHARDED_MEMORY_CONFIG` for output ← **THIS IS THE PROBLEM**

---

## Key Discovery: Wrong Code Path

**The stress test uses sharded output, but the CI deadlock is in the non-sharded output path.**

In the writer kernel (`reader_bmm_tile_layout_in1_receiver_writer_padding.cpp`):
```cpp
#ifndef OUT_SHARDED
    // WRITER — cb_wait_front(cb_id_out0, ...) + noc_async_write + cb_pop_front
    // ^^^ This is where BRISC deadlocks in CI
#endif
```

With `L1_BLOCK_SHARDED_MEMORY_CONFIG`, `OUT_SHARDED` is defined, and the entire output
writer loop is **compiled out**. BRISC never calls `cb_wait_front(output)` — the exact
function where it deadlocks in CI.

### Evidence from DPRINT analysis

Across 4 runs with progressively refined DPRINT instrumentation:

1. **in1 contention**: 54% of the time, in1 data is not ready when BRISC checks (confirmed
   across all runs). in1 multicast is consistently the slower path.

2. **in0 contention**: Nearly 0% with periodic sampling, but event-driven logging revealed
   100% contention at bw=0 (cold start). Zero contention at bw>0 (pipeline warm).

3. **All contention is at bw=0**: The first output column of every matmul has 100%
   contention on both in0 and in1. After bw=0, the pipeline is warm and data arrives
   before the receiver checks.

4. **Sender contention**: 2,751 `NC:SEND_WAIT` events, ALL at blk=7/8 bw=0, with 98.5%
   showing sem=0/7 (zero receivers acknowledged). The sender finishes the cold-start
   phase before any receiver has started.

5. **Zero output CB contention**: `BR:wait out` never fires because the output writer
   code is compiled out for sharded output.

### DPRINT infrastructure lessons learned

- **Static counters reset per kernel launch / trace replay** — `% N` modulo sampling
  only fires on the first iteration of every matmul (useless for mid-execution state).
- **Event-driven DPRINT works well** — printing only on contention (sem=0) gives
  focused, actionable data.
- **DPRINT volume matters** — early attempts with high sampling rates across all cores
  and all RISCs crashed the system. Final config: 1 core (7,7), BR+NC only, event-driven.
- **DPRINT changes timing** — it's a Heisenbug. The instrumentation overhead may prevent
  the exact race condition from occurring.

---

## What Needs to Be Done Next

### 1. Change the stress test to use non-sharded output (HIGH PRIORITY)

In `test_matmul_deadlock_stress.py`, change the matmul output memory config from:
```python
memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
```
to:
```python
memory_config=ttnn.L1_MEMORY_CONFIG  # interleaved, non-sharded
```

This will compile in the `#ifndef OUT_SHARDED` writer path where BRISC does
`cb_wait_front(output)` — the exact deadlock point from CI.

Also ensure the bias path is exercised: the QKV, self_output, and FF2 configs currently
have `fused_activation: None`. Check if the actual ViT model uses fused bias on these
matmuls (the `FUSE_BIAS` compile flag). If so, add bias tensors to the test.

### 2. Run WITHOUT DPRINT first

The DPRINT instrumentation changes timing and may prevent the deadlock. First try to
reproduce the hang with:
```bash
./vit_n300/scripts/stress_test_matmul.sh --2cq-only
```
(no `--dprint` flag). Run for many iterations (remove the `break` statement or increase
iteration count).

### 3. If hang reproduces, add targeted DPRINT

If the non-sharded output config reproduces the hang, add a single DPRINT in the
`#ifndef OUT_SHARDED` writer section:
```cpp
// Only print when output CB is NOT ready (deadlock precursor)
if (cb_tiles_available(cb_id_out0) < out_subblock_tile_count) {
    DPRINT << "BR:OUT_WAIT bh=" << bh << " bw=" << bw << " sbh=" << sbh << ENDL();
}
```

### 4. Consider running the actual ViT model test

If the isolated matmul test still can't reproduce, fall back to running the actual
`vit-N300-func` test in a loop:
```bash
./vit_n300/scripts/stress_test_vit_n300.sh
```
This exercises the real model with all its data dependencies, memory configs, and
operation chaining.

### 5. Revert kernel changes before merging

Before any PR to main:
- Revert `MCAST_INPUT_BUFFERING_DEPTH` back to `2` in `matmul_utilities.hpp`
- Remove all DPRINT instrumentation from the 2 dataflow kernels
- Remove `#include "api/debug/dprint.h"` from the dataflow kernels

---

## Environment Notes

- **Architecture**: Wormhole B0 (N300, 2 chips)
- **Container**: Docker with read-only sysfs — cannot do PCIe reset from inside container.
  If `tt-smi` breaks, run `tt-smi -r` from the HOST, or restart the container.
- **Don't Ctrl+C during `tt-smi -r`** — interrupted device resets can drop the device
  off the PCIe bus, requiring host-level intervention.
- **Build required**: Kernel changes are compile-time. Run `build_metal.sh` or equivalent
  after modifying kernel `.cpp` files.

---

## File Quick Reference

| File | What it does |
|------|-------------|
| `vit_n300/tests/test_matmul_deadlock_stress.py` | Matmul stress test (needs output config fix) |
| `vit_n300/scripts/stress_test_matmul.sh` | Runner script with DPRINT support |
| `ttnn/.../matmul_utilities.hpp` | `MCAST_INPUT_BUFFERING_DEPTH` (set to 1, revert to 2) |
| `ttnn/.../reader_bmm_tile_layout_in1_receiver_writer_padding.cpp` | in1 reader + output writer (BRISC) |
| `ttnn/.../reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp` | in0 multicast sender/receiver (NCRISC) |
| `ttnn/.../compute/bmm_large_block_zm_fused_bias_activation.cpp` | Compute kernel (DPRINT removed) |
| `vit_n300/ND_failure.log` | First CI failure triage data |
| `vit_n300/ND_failure2.log` | Second CI failure triage data |
| `vit_n300/explanations/STRESS_STRATEGY.md` | Detailed deadlock mechanism analysis |
