---
name: ttnn-dataflow-patterns
description: Get guidance on TTNN reader/writer kernel patterns (data movement, NoC operations, tensor iteration). Use when writing dataflow kernels or understanding data movement patterns.
---

# TTNN Dataflow Kernel Patterns Expert

You are helping the user with reader and writer kernel patterns in TTNN.

## Kernel Roles

| Kernel | RISC-V Core | NoC | Purpose |
|--------|-------------|-----|---------|
| Reader | RISCV_0 (BRISC) | NoC0 | Fetch data from DRAM → L1 CBs |
| Writer | RISCV_1 (NCRISC) | NoC1 | Send data from L1 CBs → DRAM |

**Note**: Names reflect convention, not strict function. Writer kernels can read data too.

## Essential Includes

```cpp
#include "dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
```

## ANTI-PATTERNS (Do NOT Use)

**InterleavedAddrGenFast and InterleavedAddrGen are DEPRECATED.**

```cpp
// WRONG - Legacy pattern, do not use
InterleavedAddrGenFast<true> src_gen = {
    .bank_base_address = src_addr,
    .page_size = tile_bytes,
    .data_format = src_fmt
};
noc_async_read_tile(i, src_gen, l1_addr);  // AVOID

// WRONG - Legacy pattern, do not use
InterleavedAddrGen<true> src_gen = {
    .bank_base_address = src_addr,
    .page_size = row_bytes
};
```

**Use TensorAccessor instead** - it works for both sharded and interleaved tensors.

## TensorAccessor: The Modern Pattern

TensorAccessor is the unified API for all tensor memory access patterns.

### Host-Side Setup (Factory)

```cpp
#include <tt-metalium/tensor_accessor_args.hpp>

// Create accessor args from buffer
const auto src_accessor_args = tt::tt_metal::TensorAccessorArgs(input.buffer());
const auto dst_accessor_args = tt::tt_metal::TensorAccessorArgs(output.buffer());

// Get compile-time and runtime args
std::vector<uint32_t> reader_ct_args;
src_accessor_args.append_to(reader_ct_args);  // Appends compile-time args

// Create kernel with compile-time args
const auto reader_id = CreateKernel(program, kernel_path, all_cores,
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .compile_args = reader_ct_args
    });

// Set runtime args (buffer address passed separately)
SetRuntimeArgs(program, reader_id, core, {
    input.buffer()->address(),
    num_pages,
    start_page
});
```

### Device-Side Setup (Kernel)

```cpp
#include "api/tensor/tensor_accessor.h"

void kernel_main() {
    // 1. Define compile-time args offset
    constexpr auto src_args = TensorAccessorArgs<0>();

    // 2. Get runtime args
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t start_page = get_arg_val<uint32_t>(2);

    // 3. Get page size from CB
    constexpr uint32_t cb_id = tt::CBIndex::c_0;
    const uint32_t page_size = get_tile_size(cb_id);

    // 4. Create accessor
    const auto src = TensorAccessor(src_args, src_addr, page_size);

    // 5. Use accessor...
}
```

## Pattern 1: Simple Reader with TensorAccessor

```cpp
#include "dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

void kernel_main() {
    constexpr auto src_args = TensorAccessorArgs<0>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_id = tt::CBIndex::c_0;
    const uint32_t page_size = get_tile_size(cb_id);

    const auto src = TensorAccessor(src_args, src_addr, page_size);

    for (uint32_t i = 0; i < num_pages; ++i) {
        cb_reserve_back(cb_id, 1);
        const uint32_t l1_addr = get_write_ptr(cb_id);
        const uint64_t noc_addr = src.get_noc_addr(i);
        noc_async_read(noc_addr, l1_addr, page_size);
        noc_async_read_barrier();
        cb_push_back(cb_id, 1);
    }
}
```

## Pattern 2: Simple Writer with TensorAccessor

```cpp
#include "dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

void kernel_main() {
    constexpr auto dst_args = TensorAccessorArgs<0>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_id = tt::CBIndex::c_16;
    const uint32_t page_size = get_tile_size(cb_id);

    const auto dst = TensorAccessor(dst_args, dst_addr, page_size);

    for (uint32_t i = 0; i < num_pages; ++i) {
        cb_wait_front(cb_id, 1);
        const uint32_t l1_addr = get_read_ptr(cb_id);
        const uint64_t noc_addr = dst.get_noc_addr(i);
        noc_async_write(l1_addr, noc_addr, page_size);
        noc_async_write_barrier();
        cb_pop_front(cb_id, 1);
    }
}
```

## Pattern 3: Pages Iterator (Recommended)

The pages iterator is more efficient for sharded tensors due to internal state caching.

```cpp
#include "dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

void kernel_main() {
    constexpr auto src_args = TensorAccessorArgs<0>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_page = get_arg_val<uint32_t>(1);
    const uint32_t end_page = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id = tt::CBIndex::c_0;
    const uint32_t page_size = get_tile_size(cb_id);

    const auto src = TensorAccessor(src_args, src_addr, page_size);

    // Iterator-based access (works for both sharded and interleaved)
    for (const auto& page : src.pages(start_page, end_page)) {
        cb_reserve_back(cb_id, 1);
        const uint32_t l1_addr = get_write_ptr(cb_id);
        noc_async_read(page.noc_addr(), l1_addr, page_size);
        noc_async_read_barrier();
        cb_push_back(cb_id, 1);
    }
}
```

## Pattern 4: Multiple Tensor Accessors

When reading from multiple tensors, chain the TensorAccessorArgs offsets:

```cpp
void kernel_main() {
    // Chain compile-time arg offsets
    constexpr auto input_args = TensorAccessorArgs<0>();
    constexpr auto gamma_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

    // Get runtime args
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t beta_addr = get_arg_val<uint32_t>(2);
    const uint32_t page_size = get_arg_val<uint32_t>(3);

    // Create accessors
    const auto input = TensorAccessor(input_args, input_addr, page_size);
    const auto gamma = TensorAccessor(gamma_args, gamma_addr, page_size);
    const auto beta = TensorAccessor(beta_args, beta_addr, page_size);

    // Read gamma/beta once (they're typically 1D)
    noc_async_read(gamma.get_noc_addr(0), gamma_l1, page_size);
    noc_async_read(beta.get_noc_addr(0), beta_l1, page_size);
    noc_async_read_barrier();

    // Read input pages
    for (uint32_t i = 0; i < num_pages; ++i) {
        noc_async_read(input.get_noc_addr(i), l1_addr, page_size);
        // ...
    }
}
```

## Pattern 5: Scaler Tile Generation

For generating constant-filled tiles (no external memory access needed):

```cpp
void kernel_main() {
    const uint32_t scaler_value = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_scaler = tt::CBIndex::c_1;

    cb_reserve_back(cb_scaler, 1);
    const uint32_t l1_addr = get_write_ptr(cb_scaler);

    // Fill tile with scaler value
    volatile tt_l1_ptr uint16_t* ptr =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr);
    for (uint32_t i = 0; i < 1024; ++i) {  // 32x32 = 1024 elements
        ptr[i] = scaler_value;
    }

    cb_push_back(cb_scaler, 1);
}
```

## NoC Operations Reference

| Function | Direction | Use Case |
|----------|-----------|----------|
| `noc_async_read()` | DRAM → L1 | Read bytes from NOC address |
| `noc_async_write()` | L1 → DRAM | Write bytes to NOC address |
| `noc_async_read_barrier()` | - | Wait for all reads to complete |
| `noc_async_write_barrier()` | - | Wait for all writes to complete |

**Note**: `noc_async_read_tile()` and `noc_async_write_tile()` are legacy APIs tied to InterleavedAddrGen. Use `noc_async_read/write()` with TensorAccessor instead.

## TensorAccessor API Quick Reference

```cpp
// Get NOC address for a page
uint64_t addr = accessor.get_noc_addr(page_id);
uint64_t addr = accessor.get_noc_addr(page_id, offset);

// Iterator over pages (efficient for sharded)
for (const auto& page : accessor.pages(start, end)) {
    uint64_t addr = page.noc_addr();
    uint32_t id = page.page_id();
}

// For sharded tensors: iterate within a shard
for (const auto& page : accessor.shard_pages(shard_id)) {
    // ...
}

// Check locality (sharded only)
bool local = accessor.is_local_page(page_id);
```

## Split Reader Pattern

When operation is compute-bound or RISC-bound, split reading between cores:

```
RISCV_0 (Reader): Reads odd pages via NoC0
RISCV_1 (Writer): Reads even pages via NoC1 (despite name)
```

**Note**: NoC1 reads are slower than NoC0. Only use split reader when:
- Operation is compute-bound (not memory-bound)
- Activation data is large enough to benefit

## Common Issues

### Data Not Reaching Compute
- Check `noc_async_read_barrier()` is called before `cb_push_back()`
- Verify TensorAccessor is constructed with correct page_size
- Check page_size matches CB configuration

### Wrong Page Order
- Verify page indexing matches expected layout
- For block iteration, check loop order (NC → Ht → Wt typical)

### Performance Issues
- Use pages iterator for sharded tensors (internal state caching)
- Consider double-buffering CBs for overlapped transfers
- Batch reads before barrier when possible

## Workflow

1. **Host-side**: Create `TensorAccessorArgs` from buffer, append to compile-time args
2. **Device-side**: Create `TensorAccessorArgs<offset>()`, then `TensorAccessor(args, addr, page_size)`
3. **Access pages**: Use `get_noc_addr(page_id)` or `pages()` iterator
4. **Sync properly**: Always barrier before CB push/pop
