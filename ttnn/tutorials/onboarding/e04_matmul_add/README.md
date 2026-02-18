# E04: Matmul + Add Kernels

Write custom kernels for matmul + add. This is the core exercise of the curriculum.

## Goal

Learn the full device operation pattern with custom kernels:
- Create device operation structure (operation_attributes_t, tensor_args_t)
- Implement ProgramFactory with custom kernels
- Write reader, compute, and writer kernels
- Compute: output = a @ b + c

## Reference

- `tt_metal/programming_examples/matmul/matmul_single_core/`
- `ttnn/cpp/ttnn/operations/full/`

## Key Concepts

### Circular Buffer Mechanics
- `cb_reserve_back` / `cb_push_back` - Producer side (reader pushes data)
- `cb_wait_front` / `cb_pop_front` - Consumer side (compute reads data)
- CBs synchronize the reader → compute → writer pipeline

### Tile Register Management
- Only 16 tile registers available
- Must acquire before use, release after
- `tile_regs_acquire/commit/wait/release` pattern

### NOC Programming
- `noc_async_read` moves data from DRAM to L1
- `noc_async_write` moves data from L1 to DRAM
- Always call barrier after async operations

### Three-Kernel Pattern
- **Reader** (RISCV_1): DRAM → CBs
- **Compute** (TRISC): CBs → math → output CB
- **Writer** (RISCV_0): output CB → DRAM

### Matmul Tiling
- Mt, Kt, Nt = tile counts in M, K, N dimensions
- Accumulate partial products across K dimension

## Common Pitfalls

1. **Forgetting barriers** - Always call `noc_async_*_barrier()` after NOC operations
2. **CB deadlock** - Ensure reader/compute/writer stay synchronized
3. **Tile register exhaustion** - Only 16 registers, acquire/release properly
4. **Re-init after switching ops** - Call `mm_init_short()` after binary ops in matmul loop

## Build & Test

```bash
cmake --build build -- onboarding
ttnn/tutorials/onboarding/run.sh "e04 and solution"
ttnn/tutorials/onboarding/run.sh "e04 and exercise"
```
