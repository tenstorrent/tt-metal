# SDPA (Scaled Dot-Product Attention) Implementation Analysis

## Overview

The SDPA (Scaled Dot-Product Attention) operation implements the FlashAttention-2 algorithm on Tenstorrent hardware. This fused operation computes:

```
Output = softmax(Q @ K^T / sqrt(d_k)) @ V
```

The implementation avoids materializing the full attention matrix by using online softmax with running statistics (max and sum), enabling memory-efficient attention computation for large sequence lengths.

**Program Factory Path**: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp`

**Key Features**:
- Online softmax with chunked Q, K, V processing
- Causality-aware load balancing across cores
- Double-buffered data movement for compute overlap
- Support for Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)
- Optional attention sinks for token absorption
- Sliding window attention support
- Paged attention (chunked mode) for KV cache

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Q chunk (block of query sequence tiles) |
| **Unit size** | `Sq_chunk_t x DHt` tiles for Q, iterated over K/V chunks |
| **Total units** | `B x NQH x q_num_chunks` (batch x query heads x Q chunks) |
| **Loop structure** | For each (batch, head, q_chunk): iterate over all k_chunks, accumulate output |

One work unit processes a single Q chunk against all relevant K/V chunks (all chunks for non-causal, or lower-triangular chunks for causal attention). The output accumulator and statistics (max, sum) are maintained in L1 until all K chunks are processed.

## Tensor Format and Layout

### Input Tensors

| Property | Q Tensor | K Tensor | V Tensor | Attention Mask |
|----------|----------|----------|----------|----------------|
| **Logical shape** | [B, NQH, Sq, DH] | [B, NKH, Sk, DH] | [B, NKH, Sk, DH_v] | [B, NQH/1, Sq, Sk] |
| **Dimension convention** | Batch, NumQueryHeads, SeqLen, HeadDim | Batch, NumKVHeads, SeqLen, HeadDim | Batch, NumKVHeads, SeqLen, HeadDimV | Batch, Heads, SeqQ, SeqK |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED | INTERLEAVED | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM | DRAM | DRAM | DRAM |
| **Data type** | BFLOAT16 / BFP8_B | BFLOAT16 / BFP8_B | BFLOAT16 / BFP8_B | BFP4_B (generated) |

**Notes**:
- `NQH / NKH` = number of query heads per KV head (for GQA/MQA support)
- K is read with transpose (DHt rows become columns in the matmul)
- When `use_mla=true`, V is read from the K tensor buffer with an offset
- Attention mask can broadcast across heads if `mask.shape[1] == 1`

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | [B, NQH, Sq, DH_v] |
| **Dimension convention** | Batch, NumQueryHeads, SeqLen, HeadDimV |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM / L1 (configurable) |
| **Data type** | Same as input (BFLOAT16 / BFP8_B) |

### Layout Transformations

1. **K Transpose**: K tiles are read with a transpose transformation applied during the read operation (rows become columns in memory layout for matmul compatibility)
2. **Causal Mask Generation**: For causal attention, a BFP4_B mask is generated on-the-fly in the writer kernel using diagonal tile patterns
3. **Padding**: Q and K sequences are internally padded to multiples of chunk sizes; padding tokens are masked during computation

## Data Flow Pattern

The SDPA operation follows a **nested iteration pattern** with online softmax accumulation:

```
For each phase in [0, num_phases):           // Multi-phase support for chunked prefill
  For each batch in [local_batch_start, local_batch_end):
    Read page_table (if chunked/paged attention)
    For each head in [local_nh_start, local_nh_end):
      Read attention_sink (if enabled)
      For each q_chunk in assigned_q_chunks:
        1. READER: Read Q chunk from DRAM -> cb_q_in
        2. Initialize statistics: prev_max = -inf, prev_sum = 0, out_accum = 0

        For each k_chunk in [0, k_num_chunks) where k_chunk overlaps with q_chunk (causal):
          3. READER: Read K chunk (transposed) from DRAM -> cb_k_in (double-buffered)
          4. READER: Read mask chunk from DRAM -> cb_mask_in (if provided)
             OR WRITER: Generate causal/sliding window mask -> cb_mask_in
          5. READER: Read V chunk from DRAM -> cb_v_in (double-buffered)

          6. COMPUTE: QK = matmul(Q, K^T) -> cb_qk_im
          7. COMPUTE: QK += mask (if on diagonal or using mask)
          8. COMPUTE: cur_max = max(QK, dim=-1), update with prev_max
          9. COMPUTE: QK = exp((QK - cur_max) * scale), cur_sum = partial_sum(QK)
          10. COMPUTE: out_im = matmul(QK, V) -> cb_out_im

          If k_chunk > 0:
            11. COMPUTE: rescale_factor = exp((prev_max - cur_max) * scale)
            12. COMPUTE: prev_sum *= rescale_factor
            13. COMPUTE: cur_sum += prev_sum
            14. COMPUTE: out_accum = out_accum * rescale_factor + out_im

          15. Swap prev/cur buffers

        16. COMPUTE: Final reduce for sum (matmul_reduce)
        17. COMPUTE: If attention_sink: process sink contribution to statistics
        18. COMPUTE: sum = 1/sum (reciprocal)
        19. COMPUTE: output = out_accum * sum -> cb_out
        20. WRITER: Write output chunk to DRAM
```

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_q_in | Q chunk input | Sq_chunk_t * DHt * q_buffer_factor | Sq_chunk_t * DHt | Single/Double | Reader | Compute | Q-chunk |
| c_1 | cb_k_in | K chunk input (transposed) | Sk_chunk_t * DHt * 2 | Sk_chunk_t * DHt | Double | Reader | Compute | K-chunk |
| c_2 | cb_v_in | V chunk input | Sk_chunk_t * vDHt * 2 | Sk_chunk_t * vDHt | Double | Reader | Compute | K-chunk |
| c_3 | cb_mask_in | Attention mask | Sq_chunk_t * Sk_chunk_t * 2 | Sq_chunk_t * Sk_chunk_t | Double | Reader/Writer | Compute | K-chunk |
| c_4 | cb_attention_sink | Attention sink values | Sq_chunk_t | Sq_chunk_t | Single | Reader | Compute | Head |
| c_5 | cb_identity_scale_in | Reduce scalar (1.0) | 1 | 1 | Single | Writer | Compute | Program |
| c_6 | cb_page_table | Page table for paged attention | page_table_stick_size | 1 stick | Single | Reader | Reader | Batch |
| c_7 | cb_col_identity | Column identity for reduce | 1 | 1 | Single | Writer | Compute | Program |
| c_16 | cb_out | Final output | Sq_chunk_t * vDHt | Sq_chunk_t * vDHt | Single | Compute | Writer | Q-chunk |
| c_24 | cb_qk_im | QK intermediate | Sq_chunk_t * Sk_chunk_t | Sq_chunk_t * Sk_chunk_t | Single | Compute | Compute | K-chunk |
| c_25 | cb_out_im_A | Output intermediate A | Sq_chunk_t * vDHt | Sq_chunk_t * vDHt | Single | Compute | Compute | Q-chunk |
| c_26 | cb_out_im_B | Output intermediate B | Sq_chunk_t * vDHt | Sq_chunk_t * vDHt | Single | Compute | Compute | Q-chunk |
| c_27 | cb_max_A | Running max A | Sq_chunk_t | Sq_chunk_t | Single | Compute | Compute | Q-chunk |
| c_28 | cb_max_B | Running max B | Sq_chunk_t | Sq_chunk_t | Single | Compute | Compute | Q-chunk |
| c_29 | cb_sum_A | Running sum A | Sq_chunk_t | Sq_chunk_t | Single | Compute | Compute | Q-chunk |
| c_30 | cb_sum_B | Running sum B | Sq_chunk_t | Sq_chunk_t | Single | Compute | Compute | Q-chunk |
| c_31 | cb_exp_max_diff | exp(prev_max - cur_max) | Sq_chunk_t | Sq_chunk_t | Single | Compute | Compute | K-chunk |

## Pipeline Pattern Summary

The SDPA operation employs several pipelining strategies:

1. **K/V Double Buffering**: `cb_k_in` and `cb_v_in` have 2x capacity, allowing the reader to prefetch the next K/V chunk while compute processes the current one
2. **Mask Double Buffering**: `cb_mask_in` has 2x capacity for overlapped mask reading/generation
3. **Q Single/Double Buffering**: `cb_q_in` uses single buffering when processing one Q chunk per core, double when processing multiple
4. **Statistics Ping-Pong**: `cb_max_A/B`, `cb_sum_A/B`, `cb_out_im_A/B` use alternating buffers across K iterations (not true double-buffering, but ping-pong for accumulation)

## Index Calculations

### Tensor Tile ID Calculation

The `TensorTileShape` class computes flat tile indices from 4D coordinates using row-major strides:

```cpp
// For shape [d0, d1, d2, d3]:
// strides = [d1*d2*d3, d2*d3, d3, 1]
tile_id = i0 * strides[0] + i1 * strides[1] + i2 * strides[2] + i3 * strides[3]
```

### Q Chunk Index (Balanced Parallelization)

When `BALANCED_Q_PARALLEL` is enabled for causal attention:

```cpp
if (q_iter < q_chunks_per_core / 2) {
    q_chunk = local_q_start + q_iter;           // Process low chunks first
} else {
    q_chunk = q_num_chunks - 1 - (local_q_start + back_q_iter);  // Then high chunks
}
```

This balances work: core 0 processes Q0 and Qn-1, core 1 processes Q1 and Qn-2, etc.

### Paged Attention Index Mapping

For chunked/paged attention mode:

```cpp
// Map virtual sequence tile to physical tile via page table
physical_block = page_table_ptr[virtual_block]
physical_tile_id = physical_block * block_stride + head_offset + block_offset
```

## Memory Access Patterns

### Read Pattern

| Data | Pattern | Notes |
|------|---------|-------|
| Q | Sequential tiles, row-major | Sq_chunk_t rows x DHt cols per chunk |
| K | Sequential tiles, transposed to row-major | Sk_chunk_t rows x DHt cols, stored transposed |
| V | Sequential tiles, row-major | Sk_chunk_t rows x vDHt cols |
| Mask | Row-by-row with stride | Sq_chunk_t rows, Sk_chunk_t cols, stride = Skt |
| Page Table | Single stick per batch | Read once per batch in chunked mode |

### Write Pattern

| Data | Pattern | Notes |
|------|---------|-------|
| Output | Sequential tiles, row-major | Only valid rows written (padded rows skipped) |

### Barrier Strategy

The reader uses a computed `barrier_threshold` to batch NOC reads before issuing barriers:

```cpp
barrier_threshold = ((512 / num_cores) * (1024 + 128)) / tile_bytes
```

This optimizes for NOC bandwidth utilization while preventing queue overflow.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (linearized to 1D for iteration) |
| **Grid dimensions** | `grid_size.x` x `grid_size.y` (from program_config or device default) |
| **Total cores** | `grid_size.x * grid_size.y` |
| **Work per core** | Variable: `batch_per_core * nh_per_core * q_per_core` Q chunks |
| **Load balancing** | Hierarchical: batch -> heads -> Q chunks; balanced for causal |

### Parallelization Factors

```cpp
batch_parallel_factor = min(B, num_cores)
nh_parallel_factor = min(num_cores / batch_parallel_factor, NQH)
q_parallel_factor = min(num_cores / (batch_parallel_factor * nh_parallel_factor), q_num_chunks)
```

### Per-Core Work Assignment

```cpp
core_id = x + y * grid_size.x

local_batch_start = (core_id / (nh_parallel_factor * q_parallel_factor)) * batch_per_core
local_batch_end = min(local_batch_start + batch_per_core, B)

local_nh_start = ((core_id / q_parallel_factor) % nh_parallel_factor) * nh_per_core
local_nh_end = min(local_nh_start + nh_per_core, NQH)

local_q_start = (core_id % q_parallel_factor) * q_per_core
local_q_end = min(local_q_start + q_per_core, q_num_chunks)
```

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | B | uint32_t | Batch size |
| 1 | NQH | uint32_t | Number of query heads |
| 2 | NKH | uint32_t | Number of KV heads |
| 3 | Sqt | uint32_t | Padded Q sequence length in tiles |
| 4 | Skt | uint32_t | Padded K sequence length in tiles |
| 5 | valid_Sqt | uint32_t | Valid (unpadded) Q sequence tiles |
| 6 | valid_Skt | uint32_t | Valid (unpadded) K sequence tiles |
| 7 | DHt | uint32_t | Head dimension in tiles |
| 8 | vDHt | uint32_t | V head dimension in tiles |
| 9 | Sq_chunk_t | uint32_t | Q chunk size in tiles |
| 10 | q_num_chunks | uint32_t | Total number of Q chunks |
| 11 | Sk_chunk_t | uint32_t | K chunk size in tiles |
| 12 | k_num_chunks | uint32_t | Total number of K chunks |
| 13 | num_cores | uint32_t | Total cores in grid |
| 14 | is_causal | uint32_t | 1 if causal attention |
| 15 | use_provided_mask | uint32_t | 1 if external mask provided |
| 16 | broadcast_provided_mask_heads | uint32_t | 1 if mask broadcasts across heads |
| 17 | use_padded_mask | uint32_t | 1 if padding requires masking |
| 18 | is_chunked | uint32_t | 1 if using paged attention |
| 19 | block_size_t | uint32_t | Block size in tiles for paging |
| 20 | page_table_stick_size | uint32_t | Page table entry size in bytes |
| 21 | use_attention_sink | uint32_t | 1 if using attention sinks |
| 22+ | TensorAccessorArgs | various | Accessor args for Q, K, V, mask, page_table, attention_sink |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0-11 | (similar to reader) | uint32_t | Dimension and chunk parameters |
| 12 | identity_scalar_packed | uint32_t | Packed 1.0f for reduce scalar |
| 13 | scale_val | uint32_t | Float32 scale value (bit-cast) |
| 14 | num_cores | uint32_t | Total cores in grid |
| 15 | is_causal | uint32_t | 1 if causal attention |
| 16 | use_provided_mask | uint32_t | 1 if external mask provided |
| 17 | use_padded_mask | uint32_t | 1 if padding requires masking |
| 18 | is_chunked | uint32_t | 1 if using paged attention |
| 19 | sliding_window_size | uint32_t | Size of sliding window (0 = disabled) |
| 20+ | TensorAccessorArgs | various | Accessor args for output tensor |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0-9 | (dimension params) | uint32_t | B, NQH, NKH, Skt, DHt, vDHt, chunk sizes |
| 10-15 | qk_matmul_params | uint32_t | QK matmul blocking parameters |
| 16-21 | out_matmul_params | uint32_t | Output matmul blocking parameters |
| 22 | num_cores | uint32_t | Total cores in grid |
| 23-26 | flags | uint32_t | is_causal, use_provided_mask, use_padded_mask, is_chunked |
| 27 | scale_fp32 | uint32_t | Scale value as fp32 bits |
| 28 | sliding_window_size | uint32_t | Sliding window size |
| 29 | use_attention_sink | uint32_t | 1 if using attention sinks |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | q_addr | uint32_t | Q tensor base address |
| 1 | k_addr | uint32_t | K tensor base address |
| 2 | v_addr | uint32_t | V tensor base address |
| 3 | mask_addr | uint32_t | Mask tensor base address (0 if none) |
| 4 | page_table_addr | uint32_t | Page table address (0 if not chunked) |
| 5 | attention_sink_addr | uint32_t | Attention sink address (0 if none) |
| 6 | core_id | uint32_t | This core's linear ID |
| 7 | local_batch_start | uint32_t | Start batch index for this core |
| 8 | local_batch_end | uint32_t | End batch index for this core |
| 9 | local_nh_start | uint32_t | Start head index for this core |
| 10 | local_nh_end | uint32_t | End head index for this core |
| 11 | local_q_start | uint32_t | Start Q chunk index for this core |
| 12 | local_q_end | uint32_t | End Q chunk index for this core |
| 13 | num_phases | uint32_t | Number of phases (1 or 2) |
| 14 | chunked_q_chunk_offset | uint32_t | Q chunk offset for chunked mode |
| 15 | read_offset | uint32_t | Read offset for chunked mode |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | out_addr | uint32_t | Output tensor base address |
| 1-12 | (similar to reader) | uint32_t | Core assignment and phase parameters |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | core_id | uint32_t | This core's linear ID |
| 1-6 | (core assignment) | uint32_t | Batch, head, Q chunk ranges |
| 7 | num_phases | uint32_t | Number of phases |
| 8 | chunked_q_chunk_offset | uint32_t | Q chunk offset for chunked mode |

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_interleaved | RISCV_0 | NOC0 | DRAM (Q,K,V,mask,page_table,sink) | cb_q_in, cb_k_in, cb_v_in, cb_mask_in, cb_attention_sink | Read tiles with optional transpose, padding, paging |

**File**: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp`

**Key Logic**:
- Uses `TensorAccessor` for address generation with various memory layouts
- `read_chunk_with_padding`: Handles reading chunks smaller than CB allocation, zero-pads remainder
- `read_paged_chunk_with_padding`: Uses page table for virtual-to-physical address translation
- K is read with `transpose=true` to prepare for matmul
- Barrier batching via `barrier_threshold` for NOC efficiency
- `fill_attention_sink_tiles`: Broadcasts sink value to all Q positions in chunk

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_interleaved | RISCV_1 | NOC1 | cb_out | DRAM | Write output, generate masks |

**File**: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/writer_interleaved.cpp`

**Key Logic**:
- `generate_reduce_scaler`: Creates scalar tile (1.0) for reduce operations
- `generate_bcast_col_scalar`: Creates column scalar for final reduce matmul
- `generate_mask`: Creates causal/sliding window mask tiles in BFP4_B format
- `generate_noncausal_padded_mask`: Creates vertical mask for sequence padding
- Only writes valid (non-padded) output rows

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| sdpa | RISCV_2/3/4 | N/A | cb_q_in, cb_k_in, cb_v_in, cb_mask_in | cb_out | Matmul, softmax, accumulation |

**File**: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/sdpa.cpp`

**Key Logic**:

1. **QK Matmul**: `matmul_blocks(cb_q_in, cb_k_in, cb_qk_im, ..., transpose=true)`
2. **Mask Application**: `add_block_inplace(cb_qk_im, cb_mask_in, ...)` for diagonal/sliding window
3. **Row-wise Max**: `reduce_c<PoolType::MAX>(cb_qk_im, ..., cur_max, prev_max, k_chunk > 0)`
4. **Scaled Exp + Partial Sum**: `sub_exp_block_bcast_cols_inplace<cb_qk_im, ..., scale_fp32, true>(cur_max, cur_sum)`
5. **QK @ V Matmul**: `matmul_blocks(cb_qk_im, cb_v_in, mm2_cur_out, ...)`
6. **Rescale Previous Stats** (k_chunk > 0):
   - `sub_exp_block<scale_fp32>(prev_max, cur_max, cb_exp_max_diff, ...)`
   - `mul_tiles_bcast_cols_inplace(prev_sum, cb_exp_max_diff, ...)`
   - `add_block_inplace(cur_sum, prev_sum, ...)`
   - `mul_block_bcast_cols<...>(mm2_prev_out, cb_exp_max_diff, mm2_cur_out, true/*accumulate*/)`
7. **Final Reduction**: `matmul_reduce<Sq_chunk_t>(cb_col_identity, prev_sum)`
8. **Attention Sink Processing** (if enabled): Updates statistics with sink contribution
9. **Reciprocal**: `recip_block_inplace(prev_sum, ...)`
10. **Final Normalization**: `mul_block_bcast_cols<...>(mm2_prev_out, prev_sum, cb_out, false)`

## Implementation Notes

### Scale Fusion Optimization

The scale factor `1/sqrt(d_k)` is fused into the exponential computation rather than applied as a separate multiply. This provides "free" scaling on the performance-critical `exp(x - max)` operation:

```cpp
exp_tile<true, true, scale_fp32>(j);  // exp with built-in scaling
```

### Balanced Q Parallelization

For causal attention, work is imbalanced (early Q chunks have less K to process). The `BALANCED_Q_PARALLEL` define enables a scheme where each core processes both low and high Q chunks:

```
Core 0: Q0 (1 K chunk), Qn-1 (n K chunks)
Core 1: Q1 (2 K chunks), Qn-2 (n-1 K chunks)
...
```

This achieves near-perfect load balancing with ~1.6x speedup over naive assignment.

### Matmul Subblock Configuration

Subblock dimensions are computed to maximize DST register utilization:

```cpp
dst_size = fp32_dest_acc_en ? 4 : 8  // DST register capacity
qk_out_subblock_w = min(Sk_chunk_t, dst_size)
qk_out_subblock_h = (full_row) ? min(Sq_chunk_t, dst_size / subblock_w) : 1
```

A 2x4 subblock is preferred when possible for better matmul utilization.

### Statistics Granularity

Several granularity parameters control loop unrolling and register usage:

- `STATS_GRANULARITY`: For statistics operations (max, sum)
- `SUB_EXP_GRANULARITY`: For subtract-exp operations
- `MUL_BCAST_GRANULARITY`: For multiply-broadcast operations
- `DHT_GRANULARITY`: For head-dimension operations
- `REDUCE_GRANULARITY`: For reduction operations (uses dst_size/2)

All must be powers of 2 for efficient implementation.

### Attention Sink Processing

Attention sinks provide "virtual tokens" that absorb attention probability without contributing to the output:

```cpp
// Attention sink is processed as a final K chunk that only affects statistics
cur_max = max(prev_max, attention_sink)
rescale_factor = exp(prev_max - cur_max)
prev_sum = prev_sum * rescale_factor + exp(attention_sink - cur_max)
prev_out = prev_out * rescale_factor  // Note: no V contribution from sink
```

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the Flash Attention algorithm work in the context of SDPA on Tenstorrent hardware? What is the online softmax technique?"
   **Reason**: Understanding the core algorithm for memory-efficient attention
   **Key Findings**: Online softmax uses running max/sum statistics to compute softmax without materializing the full attention matrix. Scale is fused into exp for efficiency. Statistics are rescaled when max changes between chunks.

2. **Query**: "How do Circular Buffers work in TT-Metal kernels?"
   **Reason**: Understanding the producer-consumer synchronization model
   **Key Findings**: CBs use `cb_reserve_back`/`cb_push_back` for producers and `cb_wait_front`/`cb_pop_front` for consumers. Double-buffering enables pipelining of data movement and compute.

3. **Query**: "What is the tile-based data format in TT-Metal?"
   **Reason**: Understanding the fundamental data unit for Tensix operations
   **Key Findings**: 32x32 tiles divided into 4 faces of 16x16. Matrix engine natively multiplies 16x16 faces. Tiles enable large contiguous bursts over NoC for efficient bandwidth utilization.

### Documentation References

1. **Source**: `tech_reports/FlashAttention/FlashAttention.md`
   **Reason**: Understanding the SDPA implementation strategy and optimizations
   **Key Information**: 20x speedup over baseline, causality-aware load balancing, double-buffering for K/V, pipelining of reader/compute/writer, BFP8 support for memory bandwidth reduction

2. **Source**: `tech_reports/tensor_layouts/tensor_layouts.md`
   **Reason**: Understanding tensor memory organization
   **Key Information**: Tiles are 32x32 with 16x16 faces in row-major order within tile. Interleaved memory distributes pages round-robin across banks.

3. **Source**: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp`
   **Reason**: Understanding the compute helper functions
   **Key Information**: Custom exp/recip implementations for first-column-only operations (VectorMode::C), L1 accumulation for partial sums, granular processing for DST register efficiency

4. **Source**: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp`
   **Reason**: Understanding data movement utilities
   **Key Information**: `TensorTileShape` for index calculations, `read_chunk_with_padding` for padded reads, `virtual_seq_tile_id_to_physical_tile_id` for paged attention, mask generation utilities for BFP4_B format
