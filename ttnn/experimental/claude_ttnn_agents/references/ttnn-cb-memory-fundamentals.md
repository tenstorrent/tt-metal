# CB and Memory Fundamentals

Shared reference for ttnn-factory-builder and ttnn-kernel-writer agents.

---

## Tensor Page Definitions

**Page** = smallest unit of memory for a tensor. Depends on tensor layout:

| Layout | Page Definition | Page Size (bytes) |
|--------|-----------------|-------------------|
| **ROW_MAJOR** | 1 row (stick) | width × element_size |
| **TILE_LAYOUT** | 1 tile (32×32) | ~2048 (bf16), ~4096 (fp32) |

```
ROW_MAJOR tensor [H, W]:           TILE_LAYOUT tensor [H, W]:
┌─────────────────────┐            ┌────────┬────────┬───┐
│ row 0 (1 page)      │            │ tile   │ tile   │...│
├─────────────────────┤            │ (0,0)  │ (0,1)  │   │
│ row 1 (1 page)      │            ├────────┼────────┼───┤
├─────────────────────┤            │ tile   │ tile   │...│
│ ...                 │            │ (1,0)  │ (1,1)  │   │
└─────────────────────┘            └────────┴────────┴───┘
Page = 1 stick                     Page = 1 tile (32×32)
```

---

## CB Page Size vs Tensor Page Size

**CRITICAL**: CB page_size is configured by program factory and can differ from tensor page size.

```cpp
// Program factory configures CB with ANY page_size
CircularBufferConfig cb_config(num_pages * page_size, {{cb_id, data_format}});
cb_config.set_page_size(cb_id, page_size);  // THIS defines what "1 page" means for push/pop
```

**CB operations use CB pages, not tensor pages:**
- `cb_reserve_back(cb, N)` reserves N × page_size bytes
- `cb_push_back(cb, N)` signals N pages ready for consumer
- `cb_wait_front(cb, N)` waits for N pages from producer
- `cb_pop_front(cb, N)` releases N pages

---

## CB Synchronization Invariant

**THE #1 CAUSE OF KERNEL HANGS:**

```
Producer push count MUST EQUAL Consumer wait count
```

```cpp
// DEADLOCK example:
// Factory: page_size = tile_size, CB holds ntiles_per_block pages
// Reader (WRONG):
for (stick = 0; stick < 32; stick++) {
    cb_reserve_back(cb, 1);   // Reserves 1 tile worth (2048 bytes)
    write_stick();            // Writes stick_size bytes (e.g., 64 bytes)
    cb_push_back(cb, 1);      // Pushes 1 page
}
// Compute:
cb_wait_front(cb, ntiles_per_block);  // Waits for ntiles_per_block pages
// DEADLOCK! Reader pushed 32 pages, compute waits for ntiles_per_block
```

**Correct pattern:**
```cpp
// Reader:
cb_reserve_back(cb, ntiles_per_block);  // Reserve full block
for (stick = 0; stick < 32; stick++) {
    write_stick();  // Write into reserved space
}
cb_push_back(cb, ntiles_per_block);     // Push what consumer expects

// Compute:
cb_wait_front(cb, ntiles_per_block);    // Matches reader's push count
```

---

## Tilize Data Flow Pattern

**Key relationship for tilize (ROW_MAJOR → TILE_LAYOUT):**

```
32 sticks = 1 tile-row = ntiles_per_block tiles worth of data

Memory equivalence:
32 × stick_size = ntiles_per_block × tile_size
32 × (W × elem_size) = (W/32) × (32 × 32 × elem_size)
```

**Tilize pattern:**

**Host-side** (program factory):
```cpp
// CB configured with tile-sized pages (memory convenience)
uint32_t ntiles_per_block = W / TILE_WIDTH;  // tiles per row
create_cb(c_0, program, all_cores, tile_size, ntiles_per_block, data_format);
```

**Device-side** (kernel):
```cpp
// Reader: batch 32 sticks, push ntiles_per_block pages
cb_reserve_back(c_0, ntiles_per_block);
for (stick = 0; stick < 32; stick++) {
    read_stick();  // stick_size bytes each
}
cb_push_back(c_0, ntiles_per_block);

// Compute: wait for ntiles_per_block pages, tilize to output
cb_wait_front(c_0, ntiles_per_block);
tilize_block(c_0, ntiles_per_block, c_out);
cb_pop_front(c_0, ntiles_per_block);
```

---

## Untilize Data Flow Pattern

**Key relationship for untilize (TILE_LAYOUT → ROW_MAJOR):**

```
1 tile → 32 sticks (each of width 32 elements)
```

**Untilize pattern:**

**Device-side** (writer kernel):
```cpp
// CB holds tiles, writer extracts sticks
cb_wait_front(cb_out, 1);  // Wait for 1 tile
uint32_t l1_addr = get_read_ptr(cb_out);

for (stick = 0; stick < 32; stick++) {
    write_stick_to_dram(l1_addr, output_stick_size);  // 32 × elem_size bytes
    l1_addr += output_stick_size;
}

cb_pop_front(cb_out, 1);  // Release the tile
```

---

## Kernel Include Paths

| Kernel Type | Include |
|-------------|---------|
| Dataflow (reader/writer) | `#include "api/dataflow/dataflow_api.h"` |
| Compute (basic) | `#include "compute_kernel_api/common.h"` |
| Compute with tilize helper | `#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"` |
| Compute with reduce helper | `#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers.hpp"` |
| Compute with untilize helper | `#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"` |

**Common mistake**: `#include "dataflow_api.h"` → Use `"api/dataflow/dataflow_api.h"`

For reduce helper API details, read `ttnn/cpp/ttnn/kernel_lib/reduce_helpers.hpp` directly.

---

## TensorAccessor Pattern

**Host-side** (program factory):
```cpp
#include <tt-metalium/tensor_accessor_args.hpp>

std::vector<uint32_t> compile_args = {page_size};  // Other args first
TensorAccessorArgs(*buffer).append_to(compile_args);  // Appends accessor args
```

**Device-side** (kernel):
```cpp
#include "api/dataflow/dataflow_api.h"  // CORRECT include path

constexpr uint32_t page_size = get_compile_time_arg_val(0);
constexpr auto tensor_args = TensorAccessorArgs<1>();  // Args start at index 1
const auto accessor = TensorAccessor(tensor_args, base_addr, page_size);

// Get NOC address for a page - both styles are valid:
uint64_t noc_addr = accessor.get_noc_addr(page_id);        // Method call (more common)
uint64_t noc_addr = get_noc_addr(page_id, accessor);       // Free function wrapper
noc_async_read(noc_addr, l1_addr, page_size);
```

**Note**: The free function `get_noc_addr(id, accessor)` internally calls `accessor.get_noc_addr(id)`. Both are correct.

---

## Quick Reference Table

| Scenario | CB Page Size | Reader Push | Compute Wait | Writer Pop |
|----------|-------------|-------------|--------------|------------|
| Tilize (per block) | tile_size | ntiles_per_block | ntiles_per_block | N/A |
| Untilize (per block) | tile_size | N/A | num_tiles | num_tiles |
| Reduce (streaming) | tile_size | 1 per tile | 1 per tile | 1 per tile |
| Pass-through | tile_size | N | N | N |

---

## Verification Checklist

Before running kernels, verify:

1. [ ] CB page_size in factory matches kernel's expected page semantics
2. [ ] Producer push count = Consumer wait count (for each CB)
3. [ ] For tilize: reader batches 32 sticks per ntiles_per_block push
4. [ ] For untilize: writer handles 32 sticks per tile pop
5. [ ] TensorAccessor index matches compile-time arg position
6. [ ] Include path is `"api/dataflow/dataflow_api.h"`
