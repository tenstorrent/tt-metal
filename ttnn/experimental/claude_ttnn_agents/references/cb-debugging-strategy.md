# Comprehensive Circular Buffer Debugging Strategy

## Overview

Circular buffer (CB) bugs fall into several categories, each requiring different detection approaches:

| Bug Type | Description | Detection Method |
|----------|-------------|------------------|
| **Missing primitive** | cb_wait without cb_pop | Static count comparison |
| **Loop imbalance** | N acquires, M releases (N≠M) per iteration | DPRINT counters + loop analysis |
| **Conditional leak** | Release in some branches but not others | Control flow analysis + DPRINT |
| **Count mismatch** | wait(N) followed by pop(M) where N≠M | Static analysis + DPRINT verification |
| **Cross-kernel mismatch** | Producer pushes X, consumer expects Y | Multi-kernel DPRINT correlation |

## Strategy 1: Balance Equation Analysis

### Principle

For any CB, over any execution path, this equation must hold:

```
Producer: Σ(reserve_back) == Σ(push_back)
Consumer: Σ(wait_front) == Σ(pop_front)
Cross-kernel: Σ(push_back by producer) == Σ(wait_front by consumer)
```

### Static Analysis Steps

```bash
# Step 1: Extract all CB operations with context
grep -n "cb_reserve_back\|cb_push_back\|cb_wait_front\|cb_pop_front" kernel.cpp

# Step 2: For each CB ID, build operation table
# Example output to analyze:
# 176: cb_wait_front(curr_in_cb_id, 1);    <- inside loop
# 234: cb_pop_front(curr_in_cb_id, 1);     <- inside loop (MISSING = bug)
# 320: cb_pop_front(curr_scalar_cb_id, 1); <- different CB

# Step 3: Identify loop boundaries containing CB ops
grep -n "for\|while" kernel.cpp  # Find loops
# Then manually verify: are acquire/release BOTH inside the same loop?
```

### Warning Signs

1. **CB operation count mismatch** - Different counts for wait vs pop
2. **Asymmetric loop placement** - wait inside loop, pop outside (or vice versa)
3. **Different iteration counts** - wait in loop(N), pop in loop(M) where N≠M

## Strategy 2: DPRINT Counter Instrumentation

### Principle

Insert counters that track cumulative CB operations and compare at key points.

### Implementation Pattern

```cpp
// Add at top of kernel (after includes)
#ifdef DEBUG_CB
uint32_t g_reserve_count = 0;
uint32_t g_push_count = 0;
uint32_t g_wait_count = 0;
uint32_t g_pop_count = 0;
#endif

// Wrap each CB operation with counter increment
#ifdef DEBUG_CB
    g_wait_count++;
    DPRINT << "WAIT #" << g_wait_count << " cb=" << cb_id << ENDL();
#endif
cb_wait_front(cb_id, num_pages);

// ... process ...

#ifdef DEBUG_CB
    g_pop_count++;
    DPRINT << "POP #" << g_pop_count << " cb=" << cb_id << ENDL();
#endif
cb_pop_front(cb_id, num_pages);

// At end of kernel or key checkpoints
#ifdef DEBUG_CB
DPRINT << "BALANCE wait=" << g_wait_count << " pop=" << g_pop_count << ENDL();
if (g_wait_count != g_pop_count) {
    DPRINT << "!!! CB LEAK DETECTED !!!" << ENDL();
}
#endif
```

### Per-Iteration Balance Check

For loop-based leaks, check balance at each iteration:

```cpp
for (uint32_t i = 0; i < num_iterations; i++) {
    uint32_t iter_wait = 0, iter_pop = 0;

    // ... operations that increment iter_wait/iter_pop ...

    #ifdef DEBUG_CB
    if (iter_wait != iter_pop) {
        DPRINT << "ITER " << i << " IMBALANCE wait=" << iter_wait
               << " pop=" << iter_pop << ENDL();
    }
    #endif
}
```

## Strategy 3: CB State Monitoring

### Principle

Use `cb_pages_reservable_at_back` and `cb_pages_available_at_front` to observe CB fill level over time.

### Diagnostic Insertion Points

```cpp
// Producer side - before reserve
DPRINT << "PRE-RESERVE cb=" << cb_id
       << " reservable=" << cb_pages_reservable_at_back(cb_id, 1) << ENDL();
cb_reserve_back(cb_id, num_pages);
DPRINT << "POST-RESERVE cb=" << cb_id
       << " reservable=" << cb_pages_reservable_at_back(cb_id, 1) << ENDL();

// Consumer side - before wait
DPRINT << "PRE-WAIT cb=" << cb_id
       << " available=" << cb_pages_available_at_front(cb_id, 1) << ENDL();
cb_wait_front(cb_id, num_pages);
DPRINT << "POST-WAIT cb=" << cb_id
       << " available=" << cb_pages_available_at_front(cb_id, 1) << ENDL();
```

### Leak Detection via State Drift

```cpp
// At start of each major loop iteration
uint32_t initial_reservable = cb_pages_reservable_at_back(cb_id, 1);

// At end of iteration
uint32_t final_reservable = cb_pages_reservable_at_back(cb_id, 1);

#ifdef DEBUG_CB
if (final_reservable < initial_reservable) {
    DPRINT << "CB DRIFT: lost " << (initial_reservable - final_reservable)
           << " pages this iteration" << ENDL();
}
#endif
```

## Strategy 4: Cross-Kernel Correlation

### Principle

Compare producer output count with consumer input count using synchronized DPRINT.

### Implementation

**Producer kernel (reader):**
```cpp
uint32_t total_pushed = 0;
// ... in loop ...
cb_push_back(cb_id, num_pages);
total_pushed += num_pages;
DPRINT << "[PRODUCER] pushed=" << total_pushed << " cb=" << cb_id << ENDL();

// At end
DPRINT << "[PRODUCER] TOTAL_PUSHED=" << total_pushed << ENDL();
```

**Consumer kernel (compute):**
```cpp
uint32_t total_waited = 0;
uint32_t total_popped = 0;
// ... in loop ...
cb_wait_front(cb_id, num_pages);
total_waited += num_pages;
// ... process ...
cb_pop_front(cb_id, num_pages);
total_popped += num_pages;
DPRINT << "[CONSUMER] waited=" << total_waited << " popped=" << total_popped << ENDL();

// At end
DPRINT << "[CONSUMER] TOTAL_WAITED=" << total_waited
       << " TOTAL_POPPED=" << total_popped << ENDL();
```

### Expected Output Analysis

```
[PRODUCER] TOTAL_PUSHED=100
[CONSUMER] TOTAL_WAITED=100 TOTAL_POPPED=100   # Healthy
[CONSUMER] TOTAL_WAITED=100 TOTAL_POPPED=50    # Leak! Missing 50 pops
[CONSUMER] TOTAL_WAITED=50 TOTAL_POPPED=50     # Deadlock! Only got 50 of 100
```

## Strategy 5: Systematic Experiment Protocol

### Phase 1: Characterize the Hang

```bash
# 1. Enable watcher
export TT_METAL_WATCHER=10

# 2. Run test
timeout 10 pytest test_file.py::test_name -v

# 3. Analyze watcher
cat generated/watcher/watcher.log | tail -50

# 4. Record:
#    - Which cores are stuck (pattern like CRBW,CRBW,W,W,W)
#    - Which CB operation type (reserve vs wait)
#    - Which kernels (k_ids)
```

### Phase 2: Instrument with Counters

```bash
# 1. Add DPRINT counter instrumentation to suspected kernel
# 2. Enable DPRINT for specific core
export TT_METAL_DPRINT_CORES="(0,0)-(0,0)"
export TT_METAL_DPRINT_RISCVS=BR,NC,TR0,TR1,TR2

# 3. Run test again
timeout 10 pytest test_file.py::test_name -v 2>&1 | tee dprint_output.txt

# 4. Analyze counter progression
grep -E "WAIT|POP|RESERVE|PUSH|BALANCE" dprint_output.txt
```

### Phase 3: Identify Imbalance Point

From DPRINT output, find where counters diverge:

```
WAIT #1 cb=5
POP #1 cb=5
WAIT #2 cb=5
POP #2 cb=5
WAIT #3 cb=5
# No POP #3 -> found the leak point!
```

### Phase 4: Correlate with Code

```bash
# Find which code path leads to the imbalance
grep -n "cb_wait_front.*5\|cb_pop_front.*5" kernel.cpp

# Check control flow around the imbalanced operation
sed -n 'START,ENDp' kernel.cpp  # Read the relevant section
```

## Quick Reference: DPRINT Instrumentation Templates

### Template 1: Basic Counter (drop-in)

```cpp
// Add after cb_wait_front calls
DPRINT << "W" << cb_id << ENDL();

// Add after cb_pop_front calls
DPRINT << "P" << cb_id << ENDL();

// Analysis: Count W vs P in output for each cb_id
```

### Template 2: CB State Snapshot

```cpp
DPRINT << "CB" << cb_id
       << " A=" << cb_pages_available_at_front(cb_id, 1)
       << " R=" << cb_pages_reservable_at_back(cb_id, 1)
       << ENDL();
```

### Template 3: Loop Iteration Tracker

```cpp
for (uint32_t i = 0; i < N; i++) {
    DPRINT << "L" << i << ENDL();
    // ... CB operations ...
    DPRINT << "/L" << i << ENDL();
}
// Analysis: Ensure each L has matching /L
```

### Template 4: Conditional Path Tracker

```cpp
if (condition) {
    DPRINT << "PATH_A" << ENDL();
    // ... operations ...
} else {
    DPRINT << "PATH_B" << ENDL();
    // ... operations ...
}
// Analysis: Which path was taken? Did both paths release?
```

## Environment Variables Reference

```bash
# Enable watcher (polls every N ms)
export TT_METAL_WATCHER=10

# Enable DPRINT for specific cores
export TT_METAL_DPRINT_CORES="(0,0)-(0,0)"     # Single core
export TT_METAL_DPRINT_CORES="(0,0)-(1,1)"     # 2x2 grid

# Enable DPRINT for specific RISC-V processors
export TT_METAL_DPRINT_RISCVS=BR              # BRISC only (reader)
export TT_METAL_DPRINT_RISCVS=NC              # NCRISC only (writer)
export TT_METAL_DPRINT_RISCVS=TR0,TR1,TR2     # All TRISCs (compute)
export TT_METAL_DPRINT_RISCVS=BR,NC,TR0,TR1,TR2  # All cores
```

## Decision Tree for CB Debugging

```
1. Test hangs?
   └─> Enable watcher, get waypoint pattern

2. Pattern is CRBW (reserve blocked)?
   └─> Consumer not releasing pages
   └─> Check: cb_pop_front count == cb_wait_front count?

3. Pattern is CWFW (wait blocked)?
   └─> Producer not providing pages
   └─> Check: cb_push_back count == cb_reserve_back count?

4. Counts match but still hangs?
   └─> Loop iteration mismatch
   └─> Add DPRINT counters inside loops
   └─> Check per-iteration balance

5. Counters show drift over iterations?
   └─> Conditional path leak
   └─> Add path tracking DPRINT
   └─> Identify which branch doesn't release

6. Cross-kernel mismatch?
   └─> Producer total != Consumer total
   └─> Add DPRINT to both kernels
   └─> Compare TOTAL_PUSHED vs TOTAL_WAITED
```

## Common Pitfalls

1. **Multi-buffering confusion**: CB may have `num_pages > 1`, ensure pop/push counts account for actual page counts, not just call counts

2. **Split reader pattern**: Two reader kernels sharing work - ensure DPRINT from both, sum their outputs

3. **Conditional compilation**: `if constexpr` paths may have different CB patterns - instrument all paths

4. **Early return**: Function returning early may skip cleanup - ensure all exit paths release resources

5. **Exception-like patterns**: TT-Metal doesn't have exceptions, but conditional error handling may skip releases
