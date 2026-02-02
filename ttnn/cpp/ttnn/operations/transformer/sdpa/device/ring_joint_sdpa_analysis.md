# Ring Joint SDPA Implementation Analysis

## Overview

**Ring Joint SDPA** (Scaled Dot-Product Attention) is a distributed multi-device implementation of attention computation that combines ring-based data gathering with joint tensor processing. This operation enables efficient attention computation across multiple devices arranged in a ring topology, where each device processes its local partition of the sequence while progressively gathering and processing key-value (KV) pairs from other devices in the ring.

**Program Factory Path**: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.cpp`

### Key Features
- **Ring All-Gather Integration**: Fuses attention computation with ring all-gather for K/V tensor distribution
- **Joint Tensor Support**: Handles both local input tensors and "joint" tensors for multi-modal attention
- **Store-and-Forward Optimization**: Uses L1-to-L1 inter-core communication for KV chunk forwarding
- **Online Softmax with LSE**: Maintains log-sum-exp (LSE) statistics for numerically stable cross-device attention aggregation

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Q chunk (block of tiles) |
| **Unit size** | `Sq_chunk_t * DHt` tiles per Q chunk |
| **Total units** | `B * NH * (num_local_q_chunks + num_joint_q_chunks)` |
| **Loop structure** | Ring iterations -> Q chunks -> KV chunks |

The operation processes attention in a hierarchical loop structure:
1. **Ring Iteration Loop**: Iterates over `ring_size` iterations, progressively processing KV chunks from different devices
2. **Q Chunk Loop**: Each core processes its assigned global Q chunks (`global_q_start` to `global_q_end`)
3. **KV Chunk Loop**: For each Q chunk, iterates over all KV chunks (local + gathered + joint on last ring iteration)

## Tensor Format and Layout

### Input Tensors

| Property | input_q/k/v | gathered_k/v | joint_q/k/v |
|----------|-------------|--------------|-------------|
| **Logical shape** | [B, NH, local_padded_N, DH] | [B, NH, padded_N, DH] | [B, NH, L, DH] |
| **Dimension convention** | BNSD (Batch, NumHeads, Seq, HeadDim) | BNSD | BNSD |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM | DRAM | DRAM |
| **Data type** | Configurable (Q: q_df, K: k_df, V: v_df) | Same as K/V | Same as Q/K/V |

**Key Dimensions**:
- `B`: Batch size
- `NH`: Number of attention heads
- `local_padded_N`: Local sequence length per device (padded_N / ring_size)
- `padded_N`: Global padded sequence length across all devices
- `L`: Joint sequence length (for multi-modal tokens)
- `DH`: Head dimension

### Output Tensors

| Property | output | joint_output | lse_output |
|----------|--------|--------------|------------|
| **Logical shape** | [B, NH, local_padded_N, DH] | [B, NH, L, DH] | [B, NH, local_padded_N+L, 1] |
| **Dimension convention** | BNSD | BNSD | BNS1 |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM | DRAM | DRAM |
| **Data type** | out_df | out_df | im_df (Float16_b) |

### Layout Transformations
- **K Transpose**: K tiles are read with `transpose=true` in the reader kernel for efficient QK matmul
- **V No Transpose**: V tiles are read without transpose
- **Padding**: Zero-padding is applied for sequence positions beyond logical boundaries

## Data Flow Pattern

### High-Level Algorithm
```
for ring_iter in [0, ring_size):
    ring_id = fused_op_receiver.get_next_ring_id_and_sync()
    do_joint_kv = (ring_id == ring_size - 1)

    for global_q_chunk in [global_q_start, global_q_end):
        read Q chunk (local or joint)

        for k_chunk in [0, num_kv_chunks):
            Read/Forward KV chunk
            Compute: QK = Q @ K^T
            Apply mask if needed
            Compute: max, exp, sum (softmax numerator)
            Compute: Out_im = softmax(QK) @ V
            Accumulate with rescaling

        if ring_iter > 0:
            Update output using LSE correction
        Write output and LSE
```

### Stage-by-Stage Data Flow

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (Q tensor) | cb_q_in (c_0) | reserve_back, push_back |
| 2 | Reader | DRAM/L1 (K tensor) | cb_k_in (c_1) | reserve_back, push_back, or receive from prev core |
| 3 | Reader | DRAM/L1 (V tensor) | cb_v_in (c_2) | reserve_back, push_back, or receive from prev core |
| 4 | Writer | N/A | cb_mask_in (c_3) | Generate mask, push_back |
| 5 | Writer | N/A | cb_scale_in, cb_identity_scale, cb_col_identity | Generate scalars |
| 6 | Compute | cb_q_in, cb_k_in | cb_qk_im (c_24) | QK matmul |
| 7 | Compute | cb_qk_im, cb_mask_in | cb_qk_im | Add mask (in-place) |
| 8 | Compute | cb_qk_im | cb_max_A/B, cb_sum_A/B | Reduce max, sub-exp-sum |
| 9 | Compute | cb_qk_im, cb_v_in | cb_out_im_A/B (c_25/26) | Softmax @ V matmul |
| 10 | Compute | Statistics CBs | cb_out_im | Rescale and accumulate |
| 11 | Compute | cb_out_im, cb_lse_in | cb_out, cb_lse_out | LSE update (ring_iter > 0) |
| 12 | Writer | cb_out (c_16), cb_lse_out (c_17) | DRAM | Write output and LSE |

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_q_in | Q input staging | `Sq_chunk_t * DHt * q_buffer_factor` tiles | Sq_chunk_t * DHt | Single/Double | Reader | Compute | Block |
| c_1 | cb_k_in | K input staging | `Sk_chunk_t * DHt * 2` tiles | Sk_chunk_t * DHt | Double | Reader | Compute | Block |
| c_2 | cb_v_in | V input staging | `Sk_chunk_t * DHt * 2` tiles | Sk_chunk_t * DHt | Double | Reader | Compute | Block |
| c_3 | cb_mask_in | Attention mask | `Sq_chunk_t * Sk_chunk_t` tiles | Sq_chunk_t * Sk_chunk_t | Single | Writer | Compute | Block |
| c_4 | cb_scale_in | Scale scalar | 1 tile | 1 tile | Single | Writer | Compute | Program |
| c_5 | cb_identity_scale_in | Identity scalar | 1 tile | 1 tile | Single | Writer | Compute | Program |
| c_6 | cb_lse_in | Previous LSE | `Sq_chunk_t` tiles | Sq_chunk_t | Single | Writer | Compute | Block |
| c_7 | cb_prev_out | Previous output | `Sq_chunk_t * DHt` tiles | Sq_chunk_t * DHt | Single | Writer | Compute | Block |
| c_8 | cb_col_identity | Column identity | 1 tile | 1 tile | Single | Writer | Compute | Program |
| c_24 | cb_qk_im | QK intermediate | `Sq_chunk_t * Sk_chunk_t` tiles | Sq_chunk_t * Sk_chunk_t | Single | Compute | Compute | Block |
| c_25 | cb_out_im_A | Output intermediate A | `Sq_chunk_t * DHt` tiles | Sq_chunk_t * DHt | Single | Compute | Compute | Block |
| c_26 | cb_out_im_B | Output intermediate B | `Sq_chunk_t * DHt` tiles | Sq_chunk_t * DHt | Single | Compute | Compute | Block |
| c_27 | cb_max_A | Current max | `Sq_chunk_t` tiles | Sq_chunk_t | Single | Compute | Compute | Block |
| c_28 | cb_max_B | Previous max | `Sq_chunk_t` tiles | Sq_chunk_t | Single | Compute | Compute | Block |
| c_29 | cb_sum_A | Current sum | `Sq_chunk_t` tiles | Sq_chunk_t | Single | Compute | Compute | Block |
| c_30 | cb_sum_B | Previous sum | `Sq_chunk_t` tiles | Sq_chunk_t | Single | Compute | Compute | Block |
| c_31 | cb_exp_max_diff | Exp(max diff) | `Sq_chunk_t` tiles | Sq_chunk_t | Single | Compute | Compute | Block |
| c_16 | cb_out | Final output | `Sq_chunk_t * DHt` tiles | Sq_chunk_t * DHt | Single | Compute | Writer | Block |
| c_17 | cb_lse_out | LSE output | `Sq_chunk_t` tiles | Sq_chunk_t | Single | Compute | Writer | Block |

## Pipeline Pattern Summary

The operation uses a combination of buffering strategies:
- **Double-buffered K/V**: Enables overlap of K/V reads with compute (capacity = 2x block)
- **Single-buffered intermediates**: Statistics and output intermediates use ping-pong aliases
- **Double-buffered Q** (conditional): When `q_per_core > 1`, Q is double-buffered

The ping-pong pattern for statistics (max_A/B, sum_A/B, out_im_A/B) enables efficient in-place updates during the KV chunk loop by swapping aliases between iterations.

## Index Calculations

### TensorAccessor Pattern
The operation uses `TensorAccessor` for address generation with the following logical shapes:

```cpp
// Local Q/K/V shape
TensorTileShape(B, NH, local_padded_Nt, DHt)

// Gathered K/V shape (full sequence across ring)
TensorTileShape(B, NH, padded_Nt, DHt)

// Joint Q/K/V shape
TensorTileShape(B, NH, Lt, DHt)

// LSE shape
TensorTileShape(B, NH, local_padded_Nt + Lt, 1)
```

### Global Q Chunk to Batch/Head/Chunk Mapping
```cpp
// Decode global_q_chunk to (batch, head, q_chunk)
nb = global_q_chunk / (NH * num_q_chunks)
nq = (global_q_chunk % (NH * num_q_chunks)) / num_q_chunks
q_chunk = global_q_chunk % num_q_chunks
is_joint_q = (q_chunk >= num_local_q_chunks)
```

### KV Chunk Ring ID Mapping
```cpp
// For ring iteration, determine which device's data to read
ring_id = fused_op_receiver.get_next_ring_id_and_sync()
kv_global_start_tile = local_padded_Nt * ring_id + k_chunk * Sk_chunk_t
```

## Memory Access Patterns

### Read Pattern
- **Q**: Sequential read per Q chunk, row-major tile order
- **K**: Sequential read with transpose flag, row-major -> column-major for matmul
- **V**: Sequential read, row-major tile order
- **Store-and-forward KV**: For chain participants, K/V chunks are received via L1-to-L1 transfer from previous core

### Write Pattern
- **Output**: Sequential write, row-major tile order
- **LSE**: Sequential write, single column
- **Store-and-forward KV**: Forward K/V chunks to next core via NoC unicast

### DRAM vs L1 Access
- **First ring iteration**: Read local K/V from DRAM
- **Subsequent ring iterations**: Read gathered K/V from DRAM (populated by AllGather)
- **Chain participants**: Receive K/V from L1 of previous core, forward to L1 of next core

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (rectangular) |
| **Grid dimensions** | `grid_size.x` x `grid_size.y` |
| **Total cores** | `num_cores = grid_size.x * grid_size.y` |
| **Work per core** | `q_per_core = ceil(all_heads_num_q_chunks / num_cores)` |
| **Load balancing** | Round-robin with remainder handling |

### Work Distribution Algorithm
```cpp
// Total work units
all_heads_num_q_chunks = B * NH * num_q_chunks
q_per_core = div_up(all_heads_num_q_chunks, num_cores)

// Per-core assignment
base_chunks_per_core = total_q_chunks / num_cores
extra_chunks = total_q_chunks % num_cores
// Core i gets: base_chunks_per_core + (i < extra_chunks ? 1 : 0)
```

### Store-and-Forward Chain Construction
For heads spanning multiple cores, a chain is constructed:
- **Injector**: First core in chain, reads KV from DRAM
- **Intermediate**: Receives KV from previous, forwards to next
- **Sink**: Last core in chain, only receives

## Arguments

### Compile-Time Arguments (Reader)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | B | uint32_t | Batch size |
| 1 | NH | uint32_t | Number of attention heads |
| 2 | DHt | uint32_t | Head dimension in tiles |
| 3 | Sq_chunk_t | uint32_t | Q chunk size in tiles |
| 4 | Sk_chunk_t | uint32_t | K chunk size in tiles |
| 5 | local_padded_N | uint32_t | Local padded sequence length |
| 6 | local_padded_Nt | uint32_t | Local padded sequence in tiles |
| 7 | padded_Nt | uint32_t | Global padded sequence in tiles |
| 8 | logical_n | uint32_t | Logical (unpadded) sequence length |
| 9 | logical_nt | uint32_t | Logical sequence in tiles |
| 10 | Lt | uint32_t | Joint sequence length in tiles |
| 11 | L | uint32_t | Joint sequence length |
| 12 | num_local_q_chunks | uint32_t | Number of local Q chunks |
| 13 | num_joint_q_chunks | uint32_t | Number of joint Q chunks |
| 14 | num_local_k_chunks | uint32_t | Number of local K chunks |
| 15 | num_joint_k_chunks | uint32_t | Number of joint K chunks |
| 16 | num_q_chunks | uint32_t | Total Q chunks per head |
| 17 | ring_size | uint32_t | Number of devices in ring |
| 18+ | TensorAccessorArgs | varies | For Q, K, V, gathered_K, gathered_V, joint_Q, joint_K, joint_V |
| N | sender_semaphore_id | uint32_t | S&F sender semaphore |
| N+1 | receiver_semaphore_id | uint32_t | S&F receiver semaphore |
| N+2 | valid_semaphore_id | uint32_t | S&F valid semaphore |

### Compile-Time Arguments (Writer)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0-16 | Same as reader | - | Dimension parameters |
| 17 | packed_identity_scalar | uint32_t | Packed bf16 identity (1.0f) |
| 18 | scale_val | uint32_t | FP32 scale as uint32_t |
| 19 | ring_size | uint32_t | Number of devices in ring |
| 20+ | TensorAccessorArgs | varies | For output, joint_output, lse_output |

### Compile-Time Arguments (Compute)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0-17 | Same as reader | - | Dimension parameters |
| 18 | qk_in0_block_w | uint32_t | QK matmul block width |
| 19 | qk_subblock_w | uint32_t | QK subblock width |
| 20 | qk_subblock_h | uint32_t | QK subblock height |
| 21 | qk_in0_num_subblocks | uint32_t | QK input0 subblocks |
| 22 | qk_in1_num_subblocks | uint32_t | QK input1 subblocks |
| 23 | qk_num_blocks | uint32_t | QK number of blocks |
| 24 | out_in0_block_w | uint32_t | Output matmul block width |
| 25 | out_subblock_w | uint32_t | Output subblock width |
| 26 | out_subblock_h | uint32_t | Output subblock height |
| 27 | out_in0_num_subblocks | uint32_t | Output input0 subblocks |
| 28 | out_in1_num_subblocks | uint32_t | Output input1 subblocks |
| 29 | out_num_blocks | uint32_t | Output number of blocks |
| 30 | scale_fp32 | uint32_t | FP32 scale value |

### Runtime Arguments (Reader)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | q_addr | uint32_t | Q tensor base address |
| 1 | k_addr | uint32_t | K tensor base address |
| 2 | v_addr | uint32_t | V tensor base address |
| 3 | gathered_k_addr | uint32_t | Gathered K base address |
| 4 | gathered_v_addr | uint32_t | Gathered V base address |
| 5 | joint_q_addr | uint32_t | Joint Q base address |
| 6 | joint_k_addr | uint32_t | Joint K base address |
| 7 | joint_v_addr | uint32_t | Joint V base address |
| 8 | global_q_start | uint32_t | Start of Q chunk range |
| 9 | global_q_end | uint32_t | End of Q chunk range |
| 10 | is_chain_participant | uint32_t | S&F chain participant flag |
| 11 | is_injector | uint32_t | S&F injector flag |
| 12 | is_sink | uint32_t | S&F sink flag |
| 13 | chain_batch | uint32_t | S&F batch index |
| 14 | chain_head | uint32_t | S&F head index |
| 15 | chain_q_chunk_start | uint32_t | S&F Q chunk start |
| 16 | chain_q_chunk_count | uint32_t | S&F Q chunk count |
| 17 | prev_physical_x | uint32_t | Previous core X coordinate |
| 18 | prev_physical_y | uint32_t | Previous core Y coordinate |
| 19 | next_physical_x | uint32_t | Next core X coordinate |
| 20 | next_physical_y | uint32_t | Next core Y coordinate |
| 21 | next_core_q_chunks | uint32_t | Next core's Q chunk count |
| 22+ | Fused op signaler args | - | Ring SDPA synchronization |

### Runtime Arguments (Writer)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | out_addr | uint32_t | Output tensor base address |
| 1 | joint_out_addr | uint32_t | Joint output base address |
| 2 | lse_addr | uint32_t | LSE output base address |
| 3 | global_q_start | uint32_t | Start of Q chunk range |
| 4 | global_q_end | uint32_t | End of Q chunk range |
| 5+ | Fused op signaler args | - | Ring SDPA synchronization |

### Runtime Arguments (Compute)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | global_q_start | uint32_t | Start of Q chunk range |
| 1 | global_q_end | uint32_t | End of Q chunk range |
| 2+ | Fused op indexer args | - | Ring ID tracking |

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| ring_joint_reader | RISCV_0 | NOC0 | DRAM (Q/K/V), L1 (S&F KV) | cb_q_in, cb_k_in, cb_v_in | Read tiles, S&F receive/forward |

**File**: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_reader.cpp`

**Key Logic**:
- Uses `RingSDPAOpReceiver` for ring synchronization with AllGather
- For each ring iteration, determines `ring_id` via `get_next_ring_id_and_sync()`
- Reads Q chunks from local or joint tensor based on `is_joint_q`
- Reads K/V chunks with conditional sources:
  - **Injector or non-participant**: Read from DRAM (local on iter 0, gathered otherwise)
  - **Chain participant**: Receive from previous core via semaphore-synchronized L1 transfer
- Forwards K/V to next core if in chain and not sink
- Uses zero-padding for positions beyond logical sequence length

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| ring_joint_writer | RISCV_1 | NOC1 | cb_out, cb_lse_out, cb_prev_out, cb_lse_in | DRAM | Write output, generate scalars/masks |

**File**: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_writer.cpp`

**Key Logic**:
- Generates scale, identity, and column identity scalars at startup
- For ring_iter > 0, reads previous output and LSE for correction
- Generates attention masks based on:
  - Global N mask (logical sequence boundary within ring iter)
  - Local N mask (local sequence padding)
  - Joint L mask (joint sequence padding)
- Writes final output and LSE to DRAM

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| ring_joint_sdpa | RISCV_2-4 | N/A | cb_q_in, cb_k_in, cb_v_in, cb_mask_in, statistics CBs | cb_out, cb_lse_out | Full SDPA computation |

**File**: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/ring_joint_sdpa.cpp`

**Key Logic**:
- Uses `RingSDPAOpIndexer` for ring iteration tracking (no semaphore waiting, just index calculation)
- Calls `sdpa_ring<>()` template function from `compute_common.hpp`
- Performs online softmax algorithm:
  1. QK = Q @ K^T (matmul with transpose)
  2. Apply mask if needed
  3. Compute running max: `cur_max = max(prev_max, row_max(QK))`
  4. Compute exp with scaling: `QK = exp((QK - cur_max) * scale)`
  5. Compute running sum: `cur_sum += row_sum(QK)`
  6. Compute partial output: `out_im = QK @ V`
  7. Rescale accumulator: `out_acc = out_acc * exp((prev_max - cur_max) * scale) + out_im`
- After all KV chunks:
  - Compute LSE: `lse = log(sum) + scale * max`
  - For ring_iter > 0, update output using sigmoid-based LSE correction:
    ```
    sig = sigmoid(cur_lse - prev_lse)
    out = prev_out - sig * (prev_out - cur_out)
    lse = prev_lse - logsigmoid(prev_lse - cur_lse)
    ```

## Implementation Notes

### Ring All-Gather Fusion
The operation fuses with `ring_attention_all_gather_async` to overlap KV gathering with attention computation. The AllGather progressively populates `gathered_input_tensor_k/v` while SDPA consumes data as it becomes available via semaphore signaling.

### Store-and-Forward Chain Optimization
For heads spanning multiple cores, a chain is constructed to avoid redundant DRAM reads:
- First core (injector) reads KV from DRAM
- Intermediate cores receive KV from L1, process, and forward
- Semaphores ensure proper synchronization:
  - `sender_semaphore`: Signaled by receiver when ready to receive
  - `receiver_semaphore`: Signaled by sender when data is written
  - `valid_semaphore`: Used to write VALID flag to receiver

### Numerically Stable LSE Aggregation
The sigmoid-based update formula ensures numerical stability when aggregating attention across ring iterations:
```
sig = sigmoid(cur_lse - prev_lse) = 1 / (1 + exp(prev_lse - cur_lse))
out = prev_out - sig * (prev_out - cur_out)
    = prev_out * (1 - sig) + cur_out * sig
    = weighted average with weights determined by relative LSEs
```

### Even KV Chunk Handling
When the number of KV chunks processed in an iteration is even, dummy CB operations are performed to maintain consistent buffer state for double-buffering:
```cpp
if (KV_chunks_processed_in_iter % 2 == 0) {
    cb_reserve_back(cb_k_in, k_chunk_tiles);
    cb_reserve_back(cb_v_in, k_chunk_tiles);
    cb_push_back(cb_k_in, k_chunk_tiles);
    cb_push_back(cb_v_in, k_chunk_tiles);
}
```

### Matmul Subblock Configuration
Subblock dimensions are computed to maximize DEST register utilization:
```cpp
dst_size = ttnn::get_dest_reg_count(compute_kernel_config)
qk_out_subblock_w = min(Sk_chunk_t, dst_size)
qk_out_subblock_h = (qk_out_subblock_w == Sk_chunk_t) ? min(Sq_chunk_t, dst_size / qk_out_subblock_w) : 1
```

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "What is Ring SDPA (Scaled Dot-Product Attention) and how does it relate to ring all-gather operations in the tt-metal architecture?"
   **Reason**: Understanding the high-level design and relationship between ring SDPA and ring all-gather
   **Key Findings**: Ring SDPA distributes attention computation across multiple devices by sharding Q along sequence dimension. Each device processes local tokens while gathering K/V via ring all-gather. The operation uses semaphores for synchronization with the AllGather fused op signaler.

2. **Query**: "How does the store-and-forward pattern work in tt-metal for inter-core communication?"
   **Reason**: Understanding the L1-to-L1 transfer mechanism used for KV forwarding
   **Key Findings**: Store-and-forward uses semaphore-based handshaking where sender waits for receiver ready signal, sends data via `noc_async_write`, then signals completion. Receiver waits for valid signal before processing.

3. **Query**: "What is TensorAccessor in tt-metal and how does it work?"
   **Reason**: Understanding how tensor indices are mapped to physical memory addresses
   **Key Findings**: TensorAccessor encapsulates tensor layout information and computes physical DRAM addresses from logical tile indices. It is configured via TensorAccessorArgs at compile-time and runtime.

4. **Query**: "How do circular buffers work in tt-metal for producer-consumer synchronization?"
   **Reason**: Understanding CB APIs used throughout the kernels
   **Key Findings**: CBs use `cb_reserve_back` (producer waits for space), `cb_push_back` (producer commits), `cb_wait_front` (consumer waits for data), `cb_pop_front` (consumer releases). These coordinate data flow between reader/compute/writer kernels.

### Documentation References

1. **Source**: `METALIUM_GUIDE.md`
   **Reason**: Core architecture and programming model concepts
   **Key Information**: Tensix core structure, tile-based computing, NoC data movement, circular buffer fundamentals

2. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: TensorAccessor utility details
   **Key Information**: Mapping logical tensor indices to physical memory locations across distributed banks

3. **Source**: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp`
   **Reason**: SDPA inner loop implementation details
   **Key Information**: `sdpa_inner_loop` template, online softmax algorithm, matmul configurations, statistics computation, LSE update formulas
