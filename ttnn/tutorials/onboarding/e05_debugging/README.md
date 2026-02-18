# E05: Debugging

Learn to debug kernels using DPRINT, Watcher, and tt-triage.

## Goal

Master kernel debugging techniques:
- Use DPRINT to inspect data inside kernels
- Use Watcher to detect illegal memory accesses and hangs
- Understand the different core types
- Use tt-triage to analyze crashes and hangs

## Reference

- `docs/source/tt-metalium/tools/kernel_print.rst`
- `docs/source/tt-metalium/tools/watcher.rst`
- `docs/source/tt-metalium/tools/tt_triage.rst`

## Key Concepts

### DPRINT Debugging
- Enable with `TT_METAL_DPRINT_CORES` environment variable
- Print values from inside kernels
- Useful for inspecting tile values, loop indices, addresses

### Watcher
- Enable with `TT_METAL_WATCHER=1`
- Detects illegal NOC transactions, L1 buffer overflows, CB misuse
- Catches errors that would otherwise cause silent corruption or hangs
- Adds runtime overhead but invaluable for debugging

### Core Types
Each Tensix core has 5 RISC-V processors:
- **RISCV_0**: Writer kernel
- **RISCV_1**: Reader kernel
- **TRISC0/1/2**: Unpack, Math, Pack

### tt-triage
- Analyzes crash dumps and hangs
- Shows core states, program counters, CB states
- Helps identify where kernel got stuck

## Common Pitfalls

1. **DPRINT slows execution** - Don't leave in production code
2. **Core type confusion** - DPRINT from wrong core won't print where expected
3. **Barrier issues** - Missing barriers cause hangs, not crashes
4. **Watcher overhead** - Disable for performance measurements
