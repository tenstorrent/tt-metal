# TTNN Python Utility Bindings Reference

Quick reference for Python-side utility functions available in `ttnn`. These mirror C++ program factory utilities, enabling Python-based program descriptors to avoid hardcoded constants.

## Buffer Query Methods (on Tensor)

The most general and portable way to get buffer geometry. These query the **actual device buffer** and work for any layout (tiled, row-major) and any dtype without manual computation.

### `tensor.buffer_page_size() -> int`

Returns the page size in bytes of the underlying device buffer.

- **Tiled tensor**: returns the tile size (e.g., 2048 for bf16, 4096 for f32)
- **Row-major tensor**: returns the stick size (width * element_size)

```python
input_tensor = ttnn.from_torch(torch.randn(32, 64), dtype=ttnn.bfloat16,
                                layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
input_tensor.buffer_page_size()   # 128  (64 elements * 2 bytes)

tiled = ttnn.from_torch(torch.randn(32, 32), dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=device)
tiled.buffer_page_size()          # 2048  (32*32 tile * 2 bytes)
```

### `tensor.buffer_aligned_page_size() -> int`

Returns the page size rounded up to the buffer's alignment requirement (DRAM or L1). This is what the NoC actually transfers per DMA operation.

```python
t = ttnn.from_torch(torch.randn(32, 17), dtype=ttnn.bfloat16,
                     layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
t.buffer_page_size()          # 34  (17 * 2 bytes)
t.buffer_aligned_page_size()  # 64  (rounded up to DRAM alignment of 32)
```

**When to use**: Reader/writer kernels need the aligned page size for correct `noc_async_read`/`noc_async_write` stride calculations when pages aren't naturally aligned.

### `tensor.buffer_num_pages() -> int`

Returns the total number of pages in the buffer.

- **Tiled tensor**: number of tiles
- **Row-major tensor**: number of sticks (rows)

```python
t = ttnn.from_torch(torch.randn(64, 128), dtype=ttnn.bfloat16,
                     layout=ttnn.TILE_LAYOUT, device=device)
t.buffer_num_pages()  # 8  (2 tile-rows * 4 tile-cols)
```

### Why these are preferred

Before these bindings, computing page/num_pages required manual layout branching:

```python
# OLD: manual, fragile, layout-specific
if layout == ttnn.TILE_LAYOUT:
    page_size = tensor.tile.get_tile_size(dtype)
    num_pages = tensor.volume() // (tile_h * tile_w)
else:
    page_size = tensor.padded_shape[-1] * tensor.element_size()
    num_pages = tensor.volume() // tensor.padded_shape[-1]
```

```python
# NEW: single call, works for any layout
page_size = tensor.buffer_page_size()
num_pages = tensor.buffer_num_pages()
```

The new approach also handles edge cases (non-standard tile shapes, padding) that manual computation may miss, since it reads directly from the allocated buffer.

---

## Math Utilities

### `ttnn.round_up(value, multiple) -> int`

Round up to the nearest multiple. Mirrors `tt::round_up` from `<tt-metalium/math.hpp>`.

```python
ttnn.round_up(100, 32)   # 128
ttnn.round_up(128, 32)   # 128
ttnn.round_up(0, 32)     # 0

# Common use: align sizes to DRAM/L1 requirements
aligned = ttnn.round_up(stick_size, ttnn.get_dram_alignment())
```

### `ttnn.div_up(numerator, denominator) -> int`

Ceiling division. Mirrors `tt::div_up` from `<tt-metalium/math.hpp>`.

```python
ttnn.div_up(100, 32)   # 4
ttnn.div_up(128, 32)   # 4
ttnn.div_up(1, 32)     # 1

# Common use: compute number of tiles
num_tiles_w = ttnn.div_up(width, 32)
```

---

## HAL Queries

Architecture-specific constants queried from hardware. Never hardcode these values.

### `ttnn.get_dram_alignment() -> int`

DRAM alignment requirement in bytes. Currently 32 on Wormhole/Blackhole.

### `ttnn.get_l1_alignment() -> int`

L1 alignment requirement in bytes. Currently 16 on Wormhole/Blackhole.

```python
dram_align = ttnn.get_dram_alignment()  # 32
l1_align = ttnn.get_l1_alignment()      # 16

# Use in CB sizing to guarantee aligned pages
aligned_page = ttnn.round_up(raw_page_size, dram_align)
```

---

## Tile Size

### `ttnn.tile_size(dtype) -> int`

Returns tile size in bytes for a standard 32x32 tile of the given dtype. Mirrors `tt::tile_size()`.

```python
ttnn.tile_size(ttnn.bfloat16)  # 2048
ttnn.tile_size(ttnn.float32)   # 4096
ttnn.tile_size(ttnn.bfloat8_b) # 1088  (256*4 + 16*4)
```

This is equivalent to `tensor.tile.get_tile_size(dtype)` but doesn't require a tensor object. Useful when computing CB sizes for intermediate buffers that have no corresponding tensor.

---

## Work Distribution

### `ttnn.find_max_divisor(val, start_max_div) -> int`

Find the largest divisor of `val` that is <= `start_max_div`, excluding 5 and 7. Mirrors `tt::tt_metal::find_max_divisor`.

```python
ttnn.find_max_divisor(32, 8)    # 8
ttnn.find_max_divisor(30, 8)    # 6  (skips 5)
ttnn.find_max_divisor(100, 10)  # 10

# Common use: find optimal block size for tiling
block_size = ttnn.find_max_divisor(num_tiles_w, 8)
```

### `ttnn.grid_to_cores(num_cores, grid_size_x, grid_size_y, row_wise=False) -> list[CoreCoord]`

Generate a list of CoreCoord objects for a grid. Column-wise by default.

```python
cores = ttnn.grid_to_cores(4, 8, 8)           # 4 cores, column-wise
cores = ttnn.grid_to_cores(4, 8, 8, True)     # 4 cores, row-wise
# Returns: [CoreCoord(0,0), CoreCoord(0,1), CoreCoord(0,2), CoreCoord(0,3)]  (column-wise)
```

### `ttnn.grid_to_cores(start, end, row_wise=False) -> list[CoreCoord]`

Generate CoreCoord list from a range.

```python
cores = ttnn.grid_to_cores(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))
# Returns 16 cores in the 4x4 range
```

---

## Usage in Program Descriptors

### Example: Tiled operation (preferred pattern)

```python
def create_program_descriptor(input_tensor, output_tensor):
    # --- Buffer geometry from tensor (layout-agnostic) ---
    page_size = input_tensor.buffer_page_size()
    num_pages = input_tensor.buffer_num_pages()
    out_page_size = output_tensor.buffer_page_size()

    # --- Work distribution ---
    grid_size = device.compute_with_storage_grid_size()
    (num_cores, all_cores, core_group_1, core_group_2,
     pages_per_core_g1, pages_per_core_g2) = ttnn.split_work_to_cores(
        grid_size, num_pages
    )

    # --- CB configuration ---
    cb_in = ttnn.CBDescriptor(
        total_size=2 * page_size,  # double buffer
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=0,
                data_format=input_tensor.dtype,
                page_size=page_size,
            )
        ],
    )
    # ... rest of descriptor
```

### Example: Row-major with alignment awareness

```python
def create_program_descriptor(input_tensor, output_tensor):
    # For row-major tensors, page_size is the stick size
    page_size = input_tensor.buffer_page_size()       # W * element_size
    aligned_page = input_tensor.buffer_aligned_page_size()  # rounded up to DRAM alignment
    num_sticks = input_tensor.buffer_num_pages()

    # Pass aligned page size to reader kernel for correct NoC transfers
    reader_ct_args = [aligned_page, page_size]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[0][0] = [
        input_tensor.buffer_address(),
        num_sticks,
        0,  # start_stick
    ]
```

### Example: Intermediate CB sizing without a tensor

```python
# When you need a CB for intermediate tiles that have no associated tensor,
# use ttnn.tile_size() to compute the page size from the dtype alone:
intermed_tile_size = ttnn.tile_size(ttnn.bfloat16)  # 2048

cb_intermed = ttnn.CBDescriptor(
    total_size=num_tiles * intermed_tile_size,
    core_ranges=core_grid,
    format_descriptors=[
        ttnn.CBFormatDescriptor(
            buffer_index=24,
            data_format=ttnn.bfloat16,
            page_size=intermed_tile_size,
        )
    ],
)
```

---

## Decision Guide: Which API to Use

| Scenario | Recommended API | Why |
|----------|----------------|-----|
| CB page size matching an input/output tensor | `tensor.buffer_page_size()` | Exact match to buffer, layout-agnostic |
| Number of work units to distribute | `tensor.buffer_num_pages()` | Exact page count from buffer |
| Reader/writer need aligned stride | `tensor.buffer_aligned_page_size()` | Accounts for DRAM/L1 padding |
| Intermediate CB (no tensor exists) | `ttnn.tile_size(dtype)` | Tile size from dtype alone |
| Manual alignment of a computed size | `ttnn.round_up(size, ttnn.get_dram_alignment())` | Explicit alignment |
| Ceiling division for tile counts | `ttnn.div_up(dim, 32)` | Cleaner than `(dim + 31) // 32` |
| Finding optimal block/chunk size | `ttnn.find_max_divisor(total, max_block)` | Avoids prime/awkward block sizes |
| Generating core coordinate lists | `ttnn.grid_to_cores(n, gx, gy)` | Matches C++ factory patterns |

### Still-valid older approaches

These approaches still work and are appropriate in some cases:

| API | When to use |
|-----|-------------|
| `tensor.tile.get_tile_size(dtype)` | When you explicitly need tile size and know layout is TILE_LAYOUT |
| `tensor.padded_shape[-1] * tensor.element_size()` | When you need the raw stick width for RM-specific math |
| `tensor.volume() // (tile_h * tile_w)` | When computing tile count from shape, not from buffer |

The buffer query methods are preferred because they work across layouts and account for padding/alignment that manual computation might miss.

---

## Summary Table

| Function | Returns | Module |
|----------|---------|--------|
| `tensor.buffer_page_size()` | Page size in bytes | Tensor method |
| `tensor.buffer_aligned_page_size()` | Aligned page size in bytes | Tensor method |
| `tensor.buffer_num_pages()` | Number of pages | Tensor method |
| `ttnn.tile_size(dtype)` | Tile size in bytes | `ttnn` |
| `ttnn.round_up(val, mult)` | Rounded-up value | `ttnn` |
| `ttnn.div_up(a, b)` | Ceiling division | `ttnn` |
| `ttnn.get_dram_alignment()` | DRAM alignment (bytes) | `ttnn` |
| `ttnn.get_l1_alignment()` | L1 alignment (bytes) | `ttnn` |
| `ttnn.find_max_divisor(val, max)` | Largest divisor | `ttnn` |
| `ttnn.grid_to_cores(n, gx, gy)` | List of CoreCoord | `ttnn` |
