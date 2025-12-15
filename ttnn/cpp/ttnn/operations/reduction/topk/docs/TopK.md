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

The TopK device operation automatically selects the appropriate strategy based on:

1. **Width Check**: Input width must be ≥ 8192
2. **K Constraint**: K must be ≤ 64
3. **Index Type**: Input dimension must be < 65536 (to use uint16 indices)
4. **Memory Verification**: L1 memory must be sufficient for:
   - Local TopK processing on each core
   - Gather phase with data from all participating cores
5. **Core Configuration**: A valid core split configuration must exist

If all conditions are met, **Multi Core** is selected. Otherwise, the operation falls back to **Single Core**.

## Strategies Description

### Single Core Strategy

The Single Core strategy processes each row of the input tensor independently on a dedicated core, using a full Bitonic Sort implementation to find the top K elements.

#### Overview:

1. **Row Assignment**:
   - Each row (spanning the full width of the tensor) is assigned to one core.
   - If there are more rows than available cores, rows are distributed such that each core processes multiple rows sequentially.

2. **Data Loading**:
   - The reader kernel loads the entire row of tiles from DRAM into circular buffers.
   - Index tiles are generated in the reader kernel to track original positions.

3. **Bitonic Sort Execution**:
   - The compute kernel performs a full Bitonic Sort on the row:
     - Tiles are transposed (since Bitonic Sort operates on columns).
     - `topk_local_sort` is used to sort pairs of tiles in-place.
     - The sort alternates between ascending and descending to build a bitonic sequence.
     - Multiple merge stages refine the ordering until the entire row is sorted.
   - The sorting respects the `largest` parameter to determine sort order.

4. **Result Extraction**:
   - After sorting, the first K elements (top K) are extracted.
   - These K elements are prepared in intermediate circular buffers.
   - Results are transposed back to the correct orientation.

5. **Output Writing**:
   - The writer kernel writes the top K values and their corresponding indices to DRAM.
   - Output consists of K tiles (rounded up from the original K request).

#### Circular Buffers:

- **c_0**: Input values (double-buffered, 4 tiles)
- **c_1**: Input indices (double-buffered, 4 tiles)
- **c_2**: Transposed values (4 tiles)
- **c_3**: Transposed indices (4 tiles)
- **c_4**: Result preparation values (2×K tiles)
- **c_5**: Result preparation indices (2×K tiles)
- **c_6**: Output values (K tiles)
- **c_7**: Output indices (K tiles)

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

The Multi Core strategy exploits parallelism by splitting the width dimension across multiple cores, each computing local top K, then gathering and computing the final top K on a dedicated core.

#### Overview:

The strategy consists of two phases:
1. **Local TopK Phase**: Each core processes a subset of the width independently.
2. **Gather and Final TopK Phase**: A final core gathers all local results and computes the global top K.

#### Phase 1: Local TopK (Split Cores)

1. **Width Splitting**:
   - The input width is divided among multiple cores.
   - Each core receives `split_size` elements (always a power of two for efficient Bitonic Sort).
   - The split is chosen such that:
     - `split_size ≥ 64` (minimum 2 tiles per core)
     - `split_size ≤ width / 2`
     - Total cores used = `width / split_size`
     - Width must divide evenly by `split_size` (no remainder)

2. **Core Configuration**:
   - Cores are arranged in a contiguous grid (e.g., x × y cores).
   - The configuration is calculated to maximize parallelism while fitting in L1 memory.

3. **Local Processing**:
   - Each core:
     - **Reads** its assigned portion of the input row (split_size elements) from DRAM.
     - **Generates or reads** corresponding indices.
     - **Executes Bitonic Sort** on its local data.
     - **Extracts local top K** elements.
     - **Sends** its local top K values and indices to the final core via NoC (L1-to-L1 communication).

4. **Communication**:
   - Each local core uses its writer kernel to send K tiles to the final core.
   - Semaphores coordinate the data transfer to ensure synchronization.

#### Phase 2: Gather and Final TopK (Final Core)

1. **Gathering**:
   - The final core receives local top K results from all split cores.
   - Data arrives via NoC directly into the final core's L1 memory.
   - Total gathered data size = `num_cores × K` elements.

2. **Final TopK Computation**:
   - The final core performs Bitonic Sort on the gathered data.
   - This produces the global top K from among all local top K results.

3. **Output Writing**:
   - The final core writes the global top K values and indices to DRAM.

#### Circular Buffers (Local Cores):

- **c_0**: Input values (double-buffered)
- **c_1**: Input indices (double-buffered)
- **c_24**: Input transposed values (Wt_local tiles)
- **c_25**: Input transposed indices (Wt_local tiles)
- **c_26**: Gathered values (Wt_final tiles, for sending)
- **c_27**: Gathered indices (Wt_final tiles, for sending)
- **c_28**: Final values (Wt_final tiles)
- **c_29**: Final indices (Wt_final tiles)
- **c_16**: Output values (double-buffered)
- **c_17**: Output indices (double-buffered)

#### Circular Buffers (Final Core):

- Same set of circular buffers
- **c_26** and **c_27** are used to receive data from local cores
- **c_28** and **c_29** are used for final Bitonic Sort and output preparation

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

### Bitonic Sort in TopK

Both strategies rely on **Bitonic Sort** as the core sorting mechanism:

1. **Bitonic Sequence Construction**:
   - Pairs of tiles are sorted alternately in ascending and descending order.
   - This creates a bitonic sequence where the first half increases then decreases (or vice versa).

2. **Bitonic Merge**:
   - Multiple stages compare and swap elements at increasing distances.
   - Each stage halves the problem size until the entire sequence is sorted.

3. **Top K Extraction**:
   - After sorting, the top K elements are the first K (if largest=True) or last K (if largest=False).
   - No additional filtering is needed; the sort naturally positions the top K at the beginning or end.

### Index Tracking

Throughout the sorting process, indices are maintained in parallel:
- Initially, indices are `[0, 1, 2, ..., n-1]`.
- Every swap operation on values is mirrored on indices.
- This ensures the final indices correctly point to the original positions of the top K elements.

### Tile-Level Operations

All operations are tile-oriented:
- `topk_local_sort`: Sorts two tiles together, updating both values and indices.
- Tiles are transposed before and after sorting since Bitonic Sort operates on columns within tiles.
- Tile boundaries are respected to maintain hardware efficiency.

---

© Tenstorrent AI ULC 2025
