# Onboarding Workshop

Workshop for kernel/op development in tt-metal.

## Architecture

```
Python API (ttnn.matmul_add)
    ↓
C++ Operation (invoke → device_operation::launch)
    ↓
Device Operation (create buffers, setup program)
    ↓
Program Factory (create CBs, configure kernels)
    ↓
Kernels (reader → compute → writer)
```

## Curriculum Overview

### Fundamentals (e01-e04)
- **e01**: Build and run tt-metal
- **e02**: TTNN basics, pytest workflow
- **e03**: Operation registration, Python bindings
- **e04**: Custom kernels (reader/compute/writer)

### Debugging & Profiling (e05-e06)
- **e05**: DPRINT debugging, tt-triage
- **e06**: Tracy profiling, peak perf calculation

### Memory & Data Layout (e07-e09)
- **e07**: L1 vs DRAM, memory banking
- **e08**: Tile layout vs row-major, data formats
- **e09**: Sharding (height, width, block)

### Scaling (e10-e11)
- **e10**: Multi-core, work splitting, multicast
- **e11**: Multi-chip, CCLs, reduction operator

### Advanced (e12-e15)
- **e12**: Matmul variants (1d, 2d), math fidelity
- **e13**: SFPU activations (gelu, relu, silu)
- **e14**: Kernel fusion (matmul + activation + CCL)
- **e15**: Pipelining, double buffering

## Key APIs

### Circular Buffers
- `cb_reserve_back(cb, n)` - Reserve n tiles for writing
- `cb_push_back(cb, n)` - Signal n tiles written
- `cb_wait_front(cb, n)` - Wait for n tiles available
- `cb_pop_front(cb, n)` - Release n tiles after reading

### Tile Registers (16 available)
- `tile_regs_acquire()` - Acquire tile registers
- `tile_regs_commit()` - Signal compute done
- `tile_regs_wait()` - Wait for commit
- `tile_regs_release()` - Release registers

### NOC (Network on Chip)
- `noc_async_read(src_addr, dst_local, size)` - DRAM → L1
- `noc_async_write(src_local, dst_addr, size)` - L1 → DRAM
- `noc_async_read_barrier()` / `noc_async_write_barrier()` - Wait for completion

### Compute Operations
- `matmul_tiles(cb_a, cb_b, a_idx, b_idx, dst_idx)` - Matrix multiply
- `add_tiles(cb_a, cb_b, a_idx, b_idx, dst_idx)` - Element-wise add
- `pack_tile(src_idx, cb_out)` - Pack from register to CB

## Common Pitfalls

1. **Forgetting barriers** - Always call `noc_async_*_barrier()` after NOC operations
2. **CB deadlock** - Ensure reader/compute/writer stay synchronized
3. **Tile register exhaustion** - Only 16 registers, acquire/release properly
4. **Re-init after switching ops** - Call `mm_init_short()` after binary ops in matmul loop

## Build Commands

```bash
./build_metal.sh                                      # Build tt-metal
cmake --build build -- onboarding                     # Build exercises
ttnn/tutorials/onboarding/run.sh "e04 and solution"  # Test
```

## Reference Code

- `ttnn/cpp/ttnn/operations/eltwise/binary/` - Simple operation pattern
- `ttnn/cpp/ttnn/operations/matmul/` - Complex device operation
- `tt_metal/programming_examples/matmul/` - Kernel examples
