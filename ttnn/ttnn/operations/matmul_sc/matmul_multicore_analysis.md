# Matmul Multicore Implementation Analysis

## Overview

The matmul multicore program factory implements batched matrix multiplication C = A * B for tiled interleaved tensors. It uses a simple tile-at-a-time compute pattern: for each output tile C[m,n], the compute kernel accumulates the dot product over all K tiles, reading one A tile and one B tile at a time from circular buffers.

**Program factory path**: `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_program_factory.cpp`

**Focus**: Compute core patterns, CB layout, runtime arg conventions, TensorAccessor setup for tiled interleaved tensors, bfloat16 data types. This analysis is intended as a reference for building a single-core tiled matmul operation (`matmul_sc`).

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile (32x32 elements) |
| **Unit size** | 1 output tile of C |
| **Total units** | `B * Mt * Nt` output tiles (batch * M-tiles * N-tiles) |
| **Loop structure** | For each output tile: inner loop over Kt to accumulate A[m,k] * B[k,n] |

One "work unit" is the computation of a single output tile. Producing it requires reading Kt pairs of (A-tile, B-tile) and accumulating the matmul results in the DST register.

## Tensor Format and Layout

### Input Tensor A (in0)

| Property | Value |
|----------|-------|
| **Logical shape** | [..., M, K] (batch dims + 2D) |
| **Dimension convention** | Last two dims are M (rows) and K (columns) |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | BFLOAT16 (or any supported; format derived from tensor dtype) |

### Input Tensor B (in1)

| Property | Value |
|----------|-------|
| **Logical shape** | [..., K, N] (batch dims + 2D) |
| **Dimension convention** | Last two dims are K (rows) and N (columns) |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | BFLOAT16 (or any supported) |

### Output Tensor C

| Property | Value |
|----------|-------|
| **Logical shape** | [..., M, N] |
| **Dimension convention** | Last two dims are M (rows) and N (columns) |
| **Tensor layout** | TILE_LAYOUT (32x32 tiles) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | BFLOAT16 (or any supported) |

### Tile Dimensions

The factory computes tile counts from padded shapes:
- `Mt = ashape[-2] / TILE_HEIGHT` -- number of tile-rows in A (and C)
- `Kt = ashape[-1] / TILE_WIDTH` -- number of tile-columns in A / tile-rows in B (inner dimension)
- `Nt = bshape[-1] / TILE_WIDTH` -- number of tile-columns in B (and C)
- `B = get_batch_size(ashape)` -- product of all batch dimensions

### Tile Size

For bfloat16 (2 bytes per element), a 32x32 tile is `32 * 32 * 2 = 2048 bytes` of payload. The actual tile size returned by `tt::tile_size(in0_data_format)` includes tile header overhead. For bfloat16, `tile_size` returns **2048 + header bytes** (typically 2080 bytes total with 32-byte header per face, though exact value depends on the format).

The factory calls `tt::tile_size(dataformat)` to get `in0_single_tile_size`, `in1_single_tile_size`, and `output_single_tile_size`. These are used to configure CB capacity.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| `c_0` (0) | cb_in0 | Input A tile staging | 2 tiles | 1 tile | Double | Reader | Compute | Block (per Kt iteration) |
| `c_1` (1) | cb_in1 | Input B tile staging | 2 tiles | 1 tile | Double | Reader | Compute | Block (per Kt iteration) |
| `c_16` (16) | cb_out | Output C tile staging | 2 tiles | 1 tile | Double | Compute | Writer | Block (per output tile) |

### CB Configuration Details

All three CBs are configured identically in structure:
```cpp
uint32_t num_input_tiles = 2;  // capacity for in0, in1
uint32_t num_output_tiles = 2; // capacity for out
```

Each CB holds **2 tiles** with a **page size of 1 tile**. This enables double-buffering: while one tile is being consumed, the producer can write the next.

**Why c_16 for output?** CB indices 0-15 are conventionally used for inputs and intermediates. Index 16 (`CBIndex::c_16`) is the conventional starting index for output CBs. This is a naming convention, not a hardware requirement.

**Named compile-time args for CB indices**: The factory passes CB indices to kernels via named compile-time args:
- Reader: `{"cb_in0", tt::CBIndex::c_0}`, `{"cb_in1", tt::CBIndex::c_1}`
- Writer: `{"cb_out", output_cb_index}` where `output_cb_index = tt::CBIndex::c_16`
- Compute: `{"cb_in0", tt::CBIndex::c_0}`, `{"cb_in1", tt::CBIndex::c_1}`, `{"cb_out", tt::CBIndex::c_16}`

Kernels retrieve these via `get_named_compile_time_arg_val("cb_in0")` etc.

## Pipeline Pattern Summary

All three CBs are double-buffered (capacity = 2 * block_size). This allows reader-compute and compute-writer overlap: the reader can push a tile into a CB while the compute kernel is consuming the previous tile from the same CB.

## Data Flow Pattern

### Step-by-step for one output tile C[m,n]

| Stage | Kernel | Action | CB Operations |
|-------|--------|--------|---------------|
| 1 | Reader | Read A[m,k] tile from DRAM into L1 | `cb_reserve_back(cb_in0, 1)` -> `noc_async_read_tile` -> `cb_push_back(cb_in0, 1)` |
| 2 | Reader | Read B[k,n] tile from DRAM into L1 | `cb_reserve_back(cb_in1, 1)` -> `noc_async_read_tile` -> `cb_push_back(cb_in1, 1)` |
| 3 | Compute | Wait for A and B tiles | `cb_wait_front(cb_in0, 1)`, `cb_wait_front(cb_in1, 1)` |
| 4 | Compute | Multiply-accumulate into DST | `matmul_tiles(cb_in0, cb_in1, 0, 0, 0)` |
| 5 | Compute | Release input tiles | `cb_pop_front(cb_in0, 1)`, `cb_pop_front(cb_in1, 1)` |
| 6 | (repeat stages 1-5 for all Kt iterations) | | |
| 7 | Compute | Pack accumulated result to output CB | `cb_reserve_back(cb_out, 1)` -> `pack_tile(0, cb_out)` -> `cb_push_back(cb_out, 1)` |
| 8 | Writer | Wait for output tile, write to DRAM | `cb_wait_front(cb_out, 1)` -> `noc_async_write_page` -> `cb_pop_front(cb_out, 1)` |

### DST Register Usage

The compute kernel uses `acquire_dst()` before the inner K loop and `release_dst()` after packing. The `matmul_tiles` call accumulates: `DST[0] += A_tile * B_tile`. After all Kt iterations, `pack_tile(0, cb_out)` moves DST register 0 to the output CB.

`dst_full_sync_en = true` in the ComputeConfig means the DST register uses full sync mode (all 16 DST registers available but with full synchronization between math and pack phases, as opposed to half-sync where math and pack can overlap on different halves of DST).

## Compute Kernel Structure (bmm.cpp) -- KEY REFERENCE

```cpp
void kernel_main() {
    constexpr int onetile = 1;

    // Compile-time args: batch, Mt, Kt, Nt
    uint32_t batch = get_compile_time_arg_val(0);
    uint32_t Mt    = get_compile_time_arg_val(1);
    uint32_t Kt    = get_compile_time_arg_val(2);
    uint32_t Nt    = get_compile_time_arg_val(3);

    // Named compile-time args for CB indices
    constexpr uint32_t cb_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t cb_in1 = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");

    // Initialize matmul hardware (unpack, math, pack engines)
    mm_init(cb_in0, cb_in1, cb_out);

    for (uint32_t nb = 0; nb < batch; nb++) {
        for (uint32_t mt_C = 0; mt_C < Mt; ++mt_C) {
            for (uint32_t nt_C = 0; nt_C < Nt; ++nt_C) {
                acquire_dst();
                for (uint32_t kt = 0; kt < Kt; kt++) {
                    cb_wait_front(cb_in0, onetile);
                    cb_wait_front(cb_in1, onetile);
                    matmul_tiles(cb_in0, cb_in1, 0, 0, 0);
                    cb_pop_front(cb_in0, onetile);
                    cb_pop_front(cb_in1, onetile);
                }
                cb_reserve_back(cb_out, onetile);
                pack_tile(0, cb_out);
                cb_push_back(cb_out, onetile);
                release_dst();
            }
        }
    }
}
```

### Key Observations for Reimplementation

1. **`mm_init(cb_in0, cb_in1, cb_out)`** must be called once before any `matmul_tiles` calls. It configures all three compute threads: unpack (sets up AB matmul unpack), math (sets up matmul math with the configured fidelity), and pack (sets up pack for the output CB).

2. **`matmul_tiles(cb_in0, cb_in1, 0, 0, 0)`** -- the three index arguments are: `in0_tile_index=0`, `in1_tile_index=0`, `dst_tile_index=0`. Since we process one tile at a time and pop after each call, tile index 0 always refers to the front tile in the CB. DST index 0 is where accumulation happens.

3. **`acquire_dst()` / `release_dst()`** bracket the accumulation of one output tile. Between acquire and release, the DST register holds partial sums. After the K loop completes, `pack_tile(0, cb_out)` moves the result out before `release_dst()`.

4. **The B, Mt, Nt loops are just nested loops that together iterate over all output tiles.** The factory comment explicitly states: "the B, Mt, Nt are just 3 for loops that technically act as 1 large loop." For multicore, B=1 and Mt=1 are set, with Nt = num_output_tiles_per_core. For single-core, you would set these to the actual batch, Mt, Nt values.

5. **MathFidelity**: Set to `HiFi4` (highest fidelity, no precision loss in FPU operations).

## Index Calculations

### Reader: Tile Index Mapping

The reader kernel maps a flat output tile index to the corresponding A and B tile indices.

Given output tile `n` (0-based within this core's work):
- Output tile's position in the output matrix: `(batch_idx, m, n_col)` where the flat output index = `batch_idx * MtNt + m * Nt + n_col`
- A tile index: `batch_idx * MtKt + m * Kt + k` (row-major in the MxK tile grid)
- B tile index: `batch_idx * KtNt + k * Nt + n_col` (row-major in the KxN tile grid)

The reader uses incremental index arithmetic rather than division/modulo:
```
itileA = output_tile_start_id / Nt * Kt  // starting row of A in tile units
itileB = output_tile_start_id % Nt       // starting column of B
```

In the inner K loop:
- `itileA += 1` (move to next K column in A's row)
- `itileB += Nt` (move to next K row in B's column, stride by Nt)

After the K loop, indices are reset for the next output tile.

### TensorAccessor Setup

**Host side** (program factory):
```cpp
// Compile-time args for reader
std::vector<uint32_t> reader_compile_time_args = {last_ktile_w, last_ktile_h};
tt::tt_metal::TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);
tt::tt_metal::TensorAccessorArgs(*src1_buffer).append_to(reader_compile_time_args);
```

`TensorAccessorArgs(buffer)` with default `ArgConfig::None` means ALL accessor parameters (rank, num_banks, tensor_shape, shard_shape, bank_coords) are passed as compile-time args. The `append_to()` call appends these to the compile-time args vector.

**Device side** (reader kernel):
```cpp
constexpr auto src0_args = TensorAccessorArgs<2>();           // starts at compile-time arg index 2
constexpr auto src1_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();
const auto s0 = TensorAccessor(src0_args, src0_addr, in0_tile_bytes);
const auto s1 = TensorAccessor(src1_args, src1_addr, in1_tile_bytes);
```

The template parameter `<2>` is the starting compile-time arg index (after `last_ktile_w` at index 0 and `last_ktile_h` at index 1). `next_compile_time_args_offset()` returns the next available index after src0's accessor args. The `TensorAccessor` constructor takes the args descriptor, the base address (runtime arg), and the page size (tile size from CB).

**Usage**: `noc_async_read_tile(tile_id, accessor, l1_addr)` reads the tile at the given flat tile index. The accessor handles bank mapping internally (for interleaved layout, it maps `tile_id` to the correct DRAM bank and offset).

**Writer side** follows the same pattern:
```cpp
// Host: TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);
// Device: constexpr auto dst_args = TensorAccessorArgs<0>();
//         const auto s = TensorAccessor(dst_args, dst_addr, page_bytes);
//         noc_async_write_page(page_id, s, l1_read_addr);
```

## Memory Access Patterns

### Read Pattern

For each output tile, the reader reads Kt tiles from A and Kt tiles from B, interleaved one-by-one:
1. Read A[m, 0], Read B[0, n]
2. Read A[m, 1], Read B[1, n]
3. ...
4. Read A[m, Kt-1], Read B[Kt-1, n]

A tiles are read sequentially along a row (consecutive tile indices). B tiles are read with stride Nt (stepping down a column in the KxN matrix).

Each read uses `noc_async_read_tile` followed immediately by `noc_async_read_barrier()` -- this is a blocking pattern (no read pipelining within the reader itself). The double-buffered CBs provide overlap between reader and compute.

### Write Pattern

The writer writes output tiles sequentially starting from `start_id`:
```
for i in [start_id, start_id + num_pages):
    wait for tile in cb_out
    noc_async_write_page(i, accessor, l1_addr)
    flush writes
    pop tile from cb_out
```

Output tiles are written in row-major order of the flattened output tensor.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from 2D grid) |
| **Grid dimensions** | Up to full compute grid |
| **Total cores** | Determined by `split_work_to_cores` |
| **Work per core** | `num_output_tiles_total / num_cores` (with remainder handling) |
| **Load balancing** | Two groups: group_1 gets `ceil` tiles, group_2 gets `floor` tiles |

**NOTE**: For a single-core implementation, this is irrelevant -- all tiles go to one core.

## Arguments

### Compute Kernel -- Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | batch | uint32_t | Number of batches (set to 1 in multicore; would be actual B for single-core) |
| 1 | Mt | uint32_t | Tile rows of output (set to 1 in multicore; actual Mt for single-core) |
| 2 | Kt | uint32_t | Inner dimension in tiles (number of A-columns / B-rows) |
| 3 | Nt | uint32_t | Tile columns of output (set to num_output_tiles_per_core in multicore; actual Nt for single-core) |

**Named compile-time args**: `cb_in0` = 0, `cb_in1` = 1, `cb_out` = 16

**Important**: The factory comment explains that B, Mt, Nt in the compute kernel are "just 3 for loops that technically act as 1 large loop." In the multicore case, B=1 and Mt=1, and Nt carries the total output tile count for that core. For a true single-core implementation, you would pass the actual B, Mt, and Nt values so the loop structure matches the mathematical definition.

### Compute Kernel Config

```cpp
tt_metal::ComputeConfig{
    .math_fidelity = MathFidelity::HiFi4,
    .dst_full_sync_en = true,
    .compile_args = {batch, Mt, Kt, Nt},
    .named_compile_args = {{"cb_in0", c_0}, {"cb_in1", c_1}, {"cb_out", c_16}}
}
```

### Reader Kernel -- Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | last_ktile_w | uint32_t | Remainder width in last K tile (for padding; 0 if aligned) |
| 1 | last_ktile_h | uint32_t | Remainder height in last K tile (always 0 here) |
| 2+ | (TensorAccessor args for src0) | uint32_t[] | Auto-appended by TensorAccessorArgs |
| N+ | (TensorAccessor args for src1) | uint32_t[] | Auto-appended by TensorAccessorArgs |

**Named compile-time args**: `cb_in0` = 0, `cb_in1` = 1

### Reader Kernel -- Runtime Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src0_addr | uint32_t | DRAM base address of tensor A |
| 1 | src1_addr | uint32_t | DRAM base address of tensor B |
| 2 | Mt | uint32_t | Tile rows of A/C |
| 3 | Kt | uint32_t | Inner dimension in tiles |
| 4 | Nt | uint32_t | Tile columns of B/C |
| 5 | MtKt | uint32_t | Precomputed Mt * Kt |
| 6 | KtNt | uint32_t | Precomputed Kt * Nt |
| 7 | batch | uint32_t | Batch size |
| 8 | bcast_B | uint32_t | Whether to broadcast B across batches (1=yes) |
| 9 | output_tile_start_id | uint32_t | First output tile index for this core |
| 10 | num_output_tiles | uint32_t | Number of output tiles this core processes |
| 11 | MtNt | uint32_t | Precomputed Mt * Nt |

### Writer Kernel -- Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0+ | (TensorAccessor args for dst) | uint32_t[] | Auto-appended by TensorAccessorArgs |

**Named compile-time args**: `cb_out` = 16

### Writer Kernel -- Runtime Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | DRAM base address of output tensor C |
| 1 | num_pages | uint32_t | Number of output tiles to write |
| 2 | start_id | uint32_t | First output tile index for this core |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_bmm_8bank_output_tiles_partitioned | RISCV_0 | NOC0 | DRAM (A, B) | CB_in0, CB_in1 | Read A,B tiles via TensorAccessor |
| bmm (compute) | RISCV_2 (unpack+math+pack) | N/A | CB_in0, CB_in1 | CB_out | mm_init, matmul_tiles, pack_tile |
| writer_unary_interleaved_start_id | RISCV_1 | NOC1 | CB_out | DRAM (C) | Write output tiles via TensorAccessor |

### Compute Kernel (bmm.cpp)
- **File**: `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm.cpp`
- **Key Logic**: Triple-nested loop (batch x Mt x Nt) with inner K-accumulation loop. Uses `mm_init` for one-time hardware configuration. Each output tile: `acquire_dst` -> K iterations of `matmul_tiles` -> `pack_tile` -> `release_dst`. Tile indices in `matmul_tiles` are always (0, 0, 0) because only one tile is in each CB at a time.

### Reader Kernel
- **File**: `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_8bank_output_tiles_partitioned.cpp`
- **Key Logic**: Uses TensorAccessor for tile reads. Incremental index arithmetic to avoid division. Handles batch broadcasting for B tensor. Includes optional K-tile padding for unaligned dimensions.

### Writer Kernel
- **File**: `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Generic single-page writer. Gets page size from CB interface. Uses `noc_async_write_page` with TensorAccessor. Calls `noc_async_writes_flushed()` after each page (not full barrier until end).

## Implementation Notes

### For Single-Core Matmul (matmul_sc) Adaptation

1. **Compute kernel can be reused almost verbatim**. Set compile-time args to actual (B, Mt, Kt, Nt) values. The reader must match the expected tile delivery order.

2. **Reader simplification**: For single-core, `output_tile_start_id = 0` and `num_output_tiles = B * Mt * Nt`. The reader's index arithmetic still works correctly. Alternatively, a simpler reader with explicit triple-nested loops (matching compute) could be clearer.

3. **CB setup is minimal**: Only 3 CBs needed (c_0, c_1, c_16), each with 2 tiles capacity. This is the simplest possible configuration.

4. **TensorAccessor is the standard way to access interleaved DRAM tensors**. On the host, create `TensorAccessorArgs(buffer)` and append to compile-time args. On the device, create `TensorAccessorArgs<offset>()` and then `TensorAccessor(args, base_addr, page_size)`. Use `noc_async_read_tile(tile_id, accessor, l1_addr)` and `noc_async_write_page(page_id, accessor, l1_addr)`.

5. **Data format handling**: Use `tt_metal::datatype_to_dataformat_converter(tensor.dtype())` to get the `DataFormat` enum, then `tt::tile_size(data_format)` to get tile size in bytes. The data format is passed to CB config.

6. **Math fidelity**: `MathFidelity::HiFi4` is the safe default (no precision loss). Can be made configurable.

7. **dst_full_sync_en = true**: Use full DST sync mode for simplicity (math and pack fully synchronize on all DST registers). This is simpler but potentially slower than half-sync for pipelined operations.

8. **Tile delivery order is critical**: The compute kernel expects tiles in the order: for each output tile (batch-major, row-major), receive Kt pairs of (A-tile, B-tile). The reader must push tiles in exactly this order.

## External Knowledge Sources

### Documentation References

1. **Source**: `tt_metal/hw/inc/api/compute/matmul.h`
   **Reason**: Understand `mm_init` and `matmul_tiles` API signatures and behavior
   **Key Information**: `mm_init(in0_cb, in1_cb, out_cb)` configures unpack, math, and pack engines. `matmul_tiles(in0_cb, in1_cb, in0_idx, in1_idx, dst_idx)` performs tile matmul and accumulates to DST.

2. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understand TensorAccessor host and device API
   **Key Information**: Host: `TensorAccessorArgs(buffer).append_to(compile_time_args)`. Device: `TensorAccessorArgs<offset>()` -> `TensorAccessor(args, addr, page_size)`. Use `noc_async_read_tile`/`noc_async_write_page` with the accessor.

3. **Source**: `tt_metal/api/tt-metalium/tensor_accessor_args.hpp`
   **Reason**: Verify host-side TensorAccessorArgs API
   **Key Information**: `append_to(vector<uint32_t>& compile_time_args)` appends all accessor params as compile-time args when using default `ArgConfig::None`.

4. **Source**: `tt_metal/hostdevcommon/api/hostdevcommon/kernel_structs.h`
   **Reason**: Verify CBIndex enum values
   **Key Information**: `c_0 = 0`, `c_1 = 1`, `c_16 = 16`. These are simple uint8 enum values.

5. **Source**: `tt_metal/hw/inc/api/compute/reg_api.h` and `tt_metal/hw/inc/api/compute/pack.h`
   **Reason**: Verify `acquire_dst`, `release_dst`, `pack_tile` signatures
   **Key Information**: `acquire_dst()` locks DST registers, `release_dst()` unlocks them. `pack_tile(dst_index, cb_id, output_tile_index=0)` packs from DST to CB.
