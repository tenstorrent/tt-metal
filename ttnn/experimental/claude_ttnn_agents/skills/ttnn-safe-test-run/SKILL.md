---
name: ttnn-safe-test-run
description: Run TTNN tests safely with hang prevention and recovery. Use before running any pytest to avoid stale device state.
---

# Safe TTNN Test Execution

Run this sequence before ANY pytest to avoid false conclusions from hung devices.

## Quick Command (Copy-Paste Ready)

```bash
pkill -9 -f pytest || true && tt-smi -r && timeout 30 pytest <test_file> -v
```

## Full Sequence (When Quick Fails)

### Step 1: Kill Hung Processes
```bash
pkill -9 -f pytest || true
```

### Step 2: Reset Device
```bash
tt-smi -r
```

### Step 3: Run with Timeout
```bash
timeout <seconds> pytest <test_file> -v
```

Choose timeout based on number of tests: ~10s per test is reasonable.

## If Test Hangs

1. Let timeout kill it (don't Ctrl+C)
2. Run full sequence again (Steps 1-3)
3. If still hangs after reset: kernel bug, not device state

## Common Pitfalls

- **Running without reset**: Device holds state from previous hung test
- **Skipping pkill**: Old pytest holds device lock
- **No timeout**: Hang blocks forever, wastes debugging time
- **Ctrl+C during hang**: Leaves device in bad state (use timeout instead)
