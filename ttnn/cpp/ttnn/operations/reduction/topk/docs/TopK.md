# TTNN TopK

## Overview

The TTNN TopK operation is a high-performance algorithm optimized for execution on Tenstorrent hardware that selects the top K largest (or smallest) elements from a tensor along a specified dimension. It is built on top of sorting techniques (leveraging Bitonic Sort internally) and is designed to work efficiently with tensor data divided into tiles.

The TopK operation is performed along a specified dimension of the input tensor, typically requiring data to be rearranged such that the TopK dimension is the innermost dimension. To maximize hardware utilization, the operation offers multiple execution strategies that leverage the parallelism available in the architecture.

## Brief Functional Description

The TTNN TopK operation returns the top K elements of an input tensor along a specified dimension. If no dimension is specified, the last dimension of the tensor is used by default.

The operation returns both:
- A tensor containing the top K values
- A tensor containing the indices representing the original positions of these values

### Arguments

- **input_tensor** (Tensor): The input tensor to extract top K elements from.
- **k** (uint32_t): The number of top elements to select.
- **dim** (int8_t, optional): The dimension along which to find top K. Defaults to -1 (last dimension).
- **largest** (bool, optional): If True, returns the K largest elements. If False, returns the K smallest. Defaults to True.
- **sorted** (bool, optional): If True, ensures the output is sorted. Defaults to True.
- **memory_config** (MemoryConfig, optional): Specifies memory configuration for the output tensors. Defaults to None.
- **sub_core_grids** (CoreRangeSet, optional): Specifies the core grid to use for computation. Defaults to all compute cores.
- **indices_tensor** (Tensor, optional): Pre-existing indices tensor to use instead of creating new ones. Defaults to None.
- **preallocated_output_tensors** (tuple of Tensors, optional): Preallocated tensors for the output values and indices. Defaults to None.

### Usage Limitations

- Supported index tensor types: `uint16`, `uint32`
- Supported value tensor types: `bfloat16`, `bfloat8_b`
- Input tensor must be in **TILE** layout
- Input shape must be 4D (after internal transformations)
- The dimension to select top K from must have at least 64 elements (min_dim_per_core)
- Combined batch dimensions (W × H × D) must be a multiple of 32
- Sharded memory configuration is not yet supported
- Multi-core implementation only supports K ≤ 64 and requires uint16 indices
- Multi-core implementation requires width ≥ 8192 (multi_core_min_width)

## Tensor Transformations

Before and after the core TopK operation, the input tensor undergoes several transformations to ensure compatibility with the underlying sorting implementation and hardware requirements. These transformations are transparent to the user and are automatically reversed after processing.

### Implementation Functions

The transformation process is implemented through two main functions in the current codebase:

- **`pre_topk_transform_tensor()`**: Prepares tensor for TopK device operation
- **`post_topk_transform_tensor()`**: Restores tensor to expected output format
- **`get_nearest_supported_k_value()`**: Rounds K to tile-aligned boundaries (multiples of 32)

### Pre-TopK Transformations

The input tensor preparation involves the following steps:

1. **Dimension Transposition**:
   - If the TopK dimension is not the last dimension, the tensor is transposed to move the TopK dimension to the last position.
   - This ensures that TopK always operates on the innermost dimension, which is required by the hardware implementation.

2. **Rank Normalization to 4D**:
   - Tensors with rank less than 4 are expanded to 4D by adding dimensions of size 1.
   - Tensors with rank greater than 4 are reshaped to 4D by combining higher dimensions.
   - This standardization simplifies the TopK implementation, as all strategies operate on 4D tensors.

3. **K Value Adjustment**:
   - K must be aligned to tile width boundaries (multiples of 32).
   - If K is not aligned, it is rounded up to the nearest supported value: `adjusted_k = tile_width × ⌈k / tile_width⌉`
   - This ensures all operations work on complete tiles.

4. **Minimum Dimension Padding**:
   - The last dimension must have at least 64 elements (2 tiles) to satisfy hardware requirements.
   - If the dimension is smaller, explicit padding is applied:
     - Padding values are `-∞` if `largest=True` (ensures padded values are not selected)
     - Padding values are `+∞` if `largest=False` (ensures padded values are not selected)

5. **Implicit Tile Padding**:
   - Tiles may have implicit padding to align with hardware tile size (typically 32×32 elements).
   - This padding is filled with appropriate sentinel values:
     - `-∞` for `largest=True` mode
     - `+∞` for `largest=False` mode

### Post-TopK Transformations

The `post_topk_transform_tensor` function reverses the pre-TopK transformations to restore the original tensor structure:

1. **Slice to Original K**:
   - If K was adjusted (rounded up), the output is sliced back to the original requested K value.
   - Both the values and indices tensors are sliced identically.

2. **Rank Restoration**:
   - Tensors are reshaped back to their original rank:
     - Tensors originally with rank < 4 are squeezed back.
     - Tensors originally with rank > 4 are reshaped to their original shape.

3. **Dimension Transpose Back**:
   - If the TopK dimension was transposed to the last position, the tensor is transposed back to restore the original dimension order.
   - Both values and indices tensors are transposed consistently.

4. **Final Logical Shape Adjustment**:
   - A final slice operation ensures the output matches the expected logical shape, accounting for any padding that was added during pre-processing.

## Strategy Comparison Overview

The TTNN TopK operation provides two execution strategies, each optimized for different tensor sizes and hardware resource constraints. The choice of strategy reflects a trade-off between **performance**, **scalability**, and **resource utilization**.

| Strategy              | Description                                                                                     | Strengths                                                                            | Weaknesses                                                                                   | Typical Use Case                                |
| --------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| **Single Core**       | Each row is processed entirely on a single core using Bitonic Sort.                            | Simple, works for any tensor size, supports both uint16 and uint32 indices.         | Limited parallelism; all work for a row happens on one core.                                 | Default strategy for most cases.                |
| **Multi Core**        | Work is split across multiple cores: local TopK on each core, then a final gather and TopK.    | **Highest parallelism**; exploits L1-to-L1 NoC communication; faster for large data. | Only supports K ≤ 64, uint16 indices only, requires width ≥ 8192, higher memory complexity. | Very wide tensors (width ≥ 8192) with small K. |

### Key Points:

* **Single Core** is the **most versatile and reliable** strategy, working for:
  * Any tensor size
  * Any K value
  * Both uint16 and uint32 index types
  * It ensures TopK can always complete, though with limited parallelism.

* **Multi Core** is the **fastest strategy for wide tensors** when applicable because it:
  * Distributes work across multiple cores for parallel processing.
  * Uses **internal core-to-core (L1-to-L1) communication** over the on-chip **Network-on-Chip (NoC)** for the gather phase.
  * However, its applicability is limited by:
    * **K ≤ 64** (one or two tiles)
    * **Width ≥ 8192** (minimum for multi-core splitting)
    * **uint16 indices only** (to fit in L1 memory constraints)
    * Sufficient **L1 memory** per core for local processing and gather phase

## Strategy Selection Logic

The TopK device operation automatically selects the appropriate strategy through a hierarchical decision process implemented in `select_program_factory()`. The selection prioritizes multi-core execution when beneficial and feasible:

### Multi-Core Prerequisites

All of the following conditions must be satisfied for multi-core execution:

1. **Width Check**: Input dimension ≥ 8192 (`multi_core_min_width`)
   - Ensures sufficient work to justify parallel execution overhead

2. **K Constraint**: K ≤ 64
   - Multi-core algorithm has optimized paths for small K values
   - Larger K values may not benefit from parallel execution

3. **Index Type**: Input dimension < 65536
   - Multi-core implementation currently only supports UInt16 indices
   - Dimensions ≥ 65536 force single-core execution with UInt32 indices

4. **Memory and Core Feasibility**: Pass `verify_multi_core_cost()` checks
   - Work must be divisible across available cores without remainder
   - Memory costs (gather + local per core) must fit within L1 cache limits
   - Contiguous rectangular core arrangement must be possible
   - Split size must meet `min_dim_per_core` requirements (≥ 64)
   - Must require genuinely multiple cores (> 1 core beneficial)

If any condition fails, the operation falls back to **Single Core** strategy.

## Strategies Description

### Single Core Strategy

The Single Core strategy processes each row of the input tensor independently on a dedicated core, using an **insertion sort with double buffering** algorithm to find the top K elements.

#### Overview:

1. **Row Assignment**:
   - Each row (spanning the full width of the tensor) is assigned to one core.
   - If there are more rows than available cores, rows are distributed such that each core processes multiple rows sequentially.

2. **Data Loading**:
   - The reader kernel loads the entire row of tiles from DRAM into circular buffers.
   - Index tiles are generated in the reader kernel to track original positions.

3. **Insertion Sort with Sliding Window**:
   - The compute kernel implements an insertion sort algorithm that maintains a sliding window of the K best elements
   - **Initialization**: Process first two input tiles (64 elements) using `topk_local_sort`
   - **Incremental Processing**: Process remaining tiles one at a time, inserting them into the maintained sorted buffer
   - **Double Buffering**: Uses result preparation buffers of size 2×K tiles for efficient insertion operations
   - **Three Processing Phases**:
     - **First Sort**: Initial processing of two tiles to establish sorted baseline
     - **Growing Phase**: Building up the sorted buffer until K elements are reached
     - **Steady State**: Buffer is full; new tiles compete with existing worst elements

4. **Buffer Management**:
   - Maintains `ktiles_saved` counter tracking valid sorted data
   - Uses adaptive buffer positioning with variable increment values
   - Alternates between buffer halves to prevent data corruption during merges

5. **Output Generation**:
   - After processing all input tiles, transpose results back from HW to WH format
   - Pack final top K values and corresponding indices to output buffers

#### Circular Buffers (Single Core):

- **input_val_cb_index**: Input values (double-buffered)
- **input_ind_cb_index**: Input indices (double-buffered)
- **transposed_val_cb_index**: Transposed values staging buffer
- **transposed_ind_cb_index**: Transposed indices staging buffer
- **result_prep_val_cb_index**: Result preparation values buffer (2×output_tiles size for double buffering)
- **result_prep_ind_cb_index**: Result preparation indices buffer (2×output_tiles size for double buffering)
- **output_val_cb_index**: Final output values (output_tiles size)
- **output_ind_cb_index**: Final output indices (output_tiles size)

#### Memory Considerations:

The Single Core strategy requires sufficient L1 memory to hold:
- Input double-buffering
- Transposed tiles
- Result preparation buffers
- Output buffers

The total memory cost is validated before execution to ensure the operation can complete.

#### Example:

For a tensor of shape `[32, 8192]` (1 × 256 tiles), k=32, single core processes one row at a time:

1. Load 256 tiles of values and generate 256 index tiles
2. Transpose tiles
3. Execute Bitonic Sort across all 256 tiles
4. Extract top 32 elements (1 tile)
5. Write results to DRAM
6. Repeat for each of the 32 rows

---

### Multi Core Strategy

The Multi Core strategy exploits parallelism by splitting the width dimension across multiple cores using a **divide-and-conquer approach with Bitonic Sort**. Each core processes its local chunk independently, then a final core performs global aggregation.

#### Overview:

The strategy consists of two phases:
1. **Local Processing Phase**: Each core processes its width chunk using bitonic sort algorithms.
2. **Global Aggregation Phase**: A final core performs bitonic merge on all local results to compute the global top K.

#### Phase 1: Local Processing (Split Cores) - `topk_local.cpp`

1. **Width Splitting**:
   - The input width is divided among multiple cores.
   - Each core receives `Wt_local` width tiles (always configured for optimal bitonic sort performance).
   - The split is chosen such that:
     - `split_size ≥ 64` (minimum 2 tiles per core)
     - `split_size ≤ width / 2`
     - Total cores used = `width / split_size`
     - Width must divide evenly by `split_size` (no remainder)

2. **Local Bitonic Sort Processing**:
   - **Initial Sort**: Process input tiles in pairs using `topk_local_sort` to create locally sorted sequences
   - **Iterative Merging**: Perform `log(Wt_local)` iterations of divide-and-conquer bitonic merging:
     - Iteration 0: Compare pairs (0,1), (2,3), (4,5) → groups of 64 elements
     - Iteration 1: Compare (0,2), (4,6), (8,10) → groups of 128 elements
     - Iteration n: Compare with distance 2^n → groups of 64*(2^(n+1)) elements
   - **Result Extraction**: Extract top Kt tiles (ceil(K/32)) containing locally optimal TopK elements

3. **Communication**:
   - Each local core sends its Kt tiles of locally optimal results to the final core
   - Uses semaphore-synchronized NoC transfers for efficient L1-to-L1 communication
   - Writer kernel manages data transmission to prevent buffer overflow

#### Phase 2: Global Aggregation (Final Core) - `topk_final.cpp`

1. **Data Gathering**:
   - Final core receives Kt tiles from each of the local cores
   - Total aggregated data: `Wt_final = num_local_cores × Kt` tiles
   - Data represents candidate TopK elements from all width chunks

2. **Global Bitonic Merge**:
   - Apply the same bitonic merge algorithm as local cores but on aggregated data
   - Perform `log(Wt_final)` iterations of bitonic merging across core boundaries
   - Produces the globally optimal TopK from all local TopK candidates

3. **Output Generation**:
   - Extract final Kt tiles containing the globally optimal TopK results
   - Transpose back to WH format and write to output DRAM buffers

#### Circular Buffers (Local Cores):

- **input_cb_index**: Input values (double-buffered)
- **index_cb_index**: Input indices (double-buffered)
- **input_transposed_cb_index**: Transposed values staging buffer (Wt_local tiles)
- **index_transposed_cb_index**: Transposed indices staging buffer (Wt_local tiles)
- **values_cb_index**: Local TopK values output (Kt tiles for transmission to final core)
- **output_ind_cb_index**: Local TopK indices output (Kt tiles for transmission to final core)

#### Circular Buffers (Final Core):

- **input_cb_index**: Received values from all local cores (Wt_final = num_cores × Kt tiles)
- **index_cb_index**: Received indices from all local cores (Wt_final = num_cores × Kt tiles)
- **input_transposed_cb_index**: Staging buffer for global bitonic merge operations (values)
- **index_transposed_cb_index**: Staging buffer for global bitonic merge operations (indices)
- **values_cb_index**: Final globally optimal TopK values output (Kt tiles)
- **output_ind_cb_index**: Final globally optimal TopK indices output (Kt tiles)

#### Synchronization:

- **Semaphores** coordinate communication between local cores and the final core.
- Each local core signals when it has sent data.
- The final core waits for all local cores before starting the final TopK computation.

#### Memory Constraints:

The Multi Core strategy is more memory-intensive because:
- Each local core needs space for its local TopK processing.
- The final core needs space for:
  - Gathering data from all local cores: `num_cores × K` elements
  - Performing final TopK on gathered data

Memory verification ensures:
```
memory_cost_local × num_cores + memory_cost_gather < L1_size × num_cores
```

This limits the maximum number of cores and the split size that can be used.

#### Example:

For a tensor of shape `[32, 16384]` (1 × 512 tiles), k=32, multi-core with 8 cores:

**Configuration:**
- `split_size = 2048` (64 tiles per core)
- 8 local cores + 1 final core = 9 cores total

**Local TopK (per core):**
1. Each core loads 64 tiles (2048 elements)
2. Executes Bitonic Sort on 64 tiles
3. Extracts top 32 elements (1 tile)
4. Sends 1 tile to final core via NoC

**Final TopK:**
1. Final core receives 8 tiles (8 × 32 = 256 elements)
2. Executes Bitonic Sort on 8 tiles
3. Extracts global top 32 elements
4. Writes 1 tile to DRAM

**Advantages:**
- 8-way parallelism in phase 1
- Fast L1-to-L1 NoC communication (no DRAM roundtrip for intermediate results)

---

## Algorithm Details

### Core Sorting Algorithms

The TopK operation uses different sorting algorithms depending on the selected strategy:

#### Single Core: Insertion Sort with Sliding Window

1. **Sliding Window Approach**:
   - Maintains a buffer of the current top K elements across all processed input
   - Processes input tiles incrementally, inserting new elements into the sorted buffer
   - Uses `topk_local_sort` for merging 64-element chunks (32 existing + 32 new)

2. **Three Processing Phases**:
   - **First Sort**: Process initial two tiles to establish baseline
   - **Growing Phase**: Expand sorted buffer until K elements are maintained
   - **Steady State**: New elements compete with existing worst elements

3. **Efficiency Characteristics**:
   - Time Complexity: O(Width × K) for insertion operations
   - Space Complexity: O(K) for maintained sorted buffer
   - Memory efficient due to streaming approach

#### Multi-Core: Divide-and-Conquer Bitonic Sort

1. **Bitonic Sequence Construction**:
   - Pairs of tiles are sorted alternately in ascending and descending order
   - Creates a bitonic sequence where sections increase then decrease (or vice versa)

2. **Bitonic Merge Iterations**:
   - Multiple stages compare and swap elements at exponentially increasing distances
   - Each iteration doubles the sequence length being compared: 64→128→256→...
   - log(Width) iterations until entire local chunk is processed

3. **Global Aggregation**:
   - Final core receives locally optimal TopK results from all local cores
   - Applies the same bitonic merge algorithm on aggregated data
   - Produces globally optimal TopK across all cores' contributions

4. **Top K Extraction**:
   - After bitonic sorting, the top K elements naturally appear in the first K positions (if largest=True) or last K (if largest=False)
   - No additional filtering needed; sort positioning handles selection

### Index Tracking

Throughout the sorting process, indices are maintained in parallel:
- Initially, indices are `[0, 1, 2, ..., n-1]`.
- Every swap operation on values is mirrored on indices.
- This ensures the final indices correctly point to the original positions of the top K elements.

### Shared Implementation Components

#### Tile-Level Operations

Both strategies share common tile-oriented operations:
- **`topk_local_sort`**: Core sorting primitive that processes two tiles (64 elements), updating both values and indices
  - **Single Core Usage**: Used for merging new input with existing sorted elements during insertion
  - **Multi-Core Usage**: Used for bitonic merge operations between tile pairs
- **Transpose Operations**: Tiles are transposed between WH and HW formats to match optimal processing layouts
- **Tile Boundaries**: All operations respect 32-element tile boundaries for hardware efficiency

---

© Tenstorrent AI ULC 2026
