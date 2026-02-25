# E05: Debugging

Learn to debug kernels using tt-triage, DPRINT, and Watcher.

## Goal

Master kernel debugging techniques:
- Use tt-triage to analyze crashes and hangs
- Use DPRINT to inspect data inside kernels
- Use Watcher to detect illegal memory accesses

## Reference

Read these first:
- `docs/source/tt-metalium/tools/tt_triage.rst` - Hang/crash analysis
- `docs/source/tt-metalium/tools/kernel_print.rst` - DPRINT usage
- `docs/source/tt-metalium/tools/watcher.rst` - Runtime error detection

## Operation

This exercise implements a sign function using a three-kernel pipeline:

```
Reader (RISCV_1) → Compute (TRISC) → Writer (RISCV_0)
     ↓                  ↓                  ↓
  DRAM → cb_in      sign(x)           cb_out → DRAM
```

The compute kernel applies `sign(x)`: returns -1, 0, or 1 based on input sign.

## Exercise

The exercise has a **built-in bug that causes a hang**. Your task:

1. Run the exercise and observe the hang
2. Use tt-triage to identify which kernel is stuck
3. Add DPRINT statements to understand what's happening
4. Find and fix the bug
5. Study the solution for DPRINT patterns

## Walkthrough

### Step 1: Run and Observe the Hang

```bash
./run.sh "e05 and exercise"
```

The test will hang indefinitely.

### Step 2: Diagnose with tt-triage

In another terminal:

```bash
tt-metal/scripts/install_debugger.sh
tt-metal/tools/tt-triage.py
```

Look for the stuck kernel in the output:

```
│ 1-1 (0,0) │ trisc1 │ ... │ compute │ ... │ 0x8fc8 │
│           │        │     │         │     │        │ #0 in chlkc_math::math_main() at
│           │        │     │         │     │        │ .../exercise_cpp/device/kernels/compute.cpp
│           │        │     │         │     │        │ 34:5
```

This tells you: **trisc1** (compute kernel) is stuck at **line 34** of `compute.cpp`.

### Step 3: Recover the Device

```bash
# In the terminal where the test is running:
Ctrl+Z          # Pause the process
kill %1         # Kill it
tt-smi -r       # Reset the device
```

### Step 4: Add DPRINT to Understand the Issue

Without a debugger, DPRINT is your primary tool for understanding kernel behavior.
Add print statements to the compute kernel to trace execution:

```cpp
#include "api/debug/dprint.h"

// Inside kernel_main():
DPRINT << "Processing tile " << i << ENDL();
```

Run with DPRINT enabled:

```bash
export TT_METAL_DPRINT_CORES=0,0
./run.sh "e05 and exercise"
```

You'll see output up to where the kernel hangs, helping you localize the issue.

### Step 5: Fix the Bug

Open `exercise_cpp/device/kernels/compute.cpp` and find line 33:

```cpp
asm volatile("ebreak");  // RISC-V breakpoint - delete this line
```

Delete or comment out this line, then run again.
Kernel will be re-compiled JIT.

### Step 6: Study the Solution

Review `solution_cpp/device/kernels/compute.cpp` for DPRINT patterns and best practices.

## Common Pitfalls

1. **DPRINT slows execution** - Remove from production code
2. **Forgetting ENDL()** - Buffer won't flush without it
3. **MATH can't access CBs** - TileSlice is invalid in DPRINT_MATH
4. **Watcher overhead** - Disable for performance measurements
