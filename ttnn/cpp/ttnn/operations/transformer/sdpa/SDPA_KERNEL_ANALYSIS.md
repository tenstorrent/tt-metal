# Scaled Dot Product Attention (SDPA) Kernel Analysis

This document provides a comprehensive technical analysis of the SDPA operation implementation in tt-metal, focusing on kernel-level details, compute phases, data flow patterns, and low-level optimizations.

## Table of Contents

1. [Overview](#overview)
2. [Operation Variants](#operation-variants)
3. [Tensor Layout and Shapes](#tensor-layout-and-shapes)
4. [Circular Buffer Configuration](#circular-buffer-configuration)
5. [Compute Kernel Deep Dive](#compute-kernel-deep-dive)
6. [Reader Kernel Analysis](#reader-kernel-analysis)
7. [Writer Kernel Analysis](#writer-kernel-analysis)
8. [Low-Level Optimizations](#low-level-optimizations)
9. [Kernel Capabilities and Limitations](#kernel-capabilities-and-limitations)
10. [Performance Considerations](#performance-considerations)

---

## Overview

The SDPA operation implements the core attention mechanism used in transformer models:

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

**File Locations:**
- Program Factory: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp`
- Compute Kernel: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/sdpa.cpp`
- Compute Helpers: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp`
- Reader Kernel: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp`
- Writer Kernel: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/writer_interleaved.cpp`
- Dataflow Helpers: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp`

---

## Operation Variants

The SDPA implementation supports multiple variants:

| Variant | Description | Key Features |
|---------|-------------|--------------|
| **Standard SDPA** | Basic scaled dot-product attention | Causal/non-causal, optional mask |
| **Chunked SDPA** | Paged attention for long sequences | Page table support, chunk_start_idx |
| **MLA (Multi-Latent Attention)** | Shared K/V with different head dimensions | `use_mla=true`, `head_dim_v` parameter |
| **Sliding Window** | Limited attention window | `sliding_window_size` parameter |
| **Attention Sink** | Absorbs attention probability | Virtual K chunk with no V contribution |
| **Joint SDPA** | Cross-attention patterns | Separate Q/K/V for joint processing |
| **Ring Distributed** | Multi-device attention | Ring topology for distributed execution |

---

## Tensor Layout and Shapes

### Input Tensors

| Tensor | Shape | Description |
|--------|-------|-------------|
| Q | `[B, NQH, Sq, DH]` | Query tensor |
| K | `[B, NKH, Sk, DH]` | Key tensor (transposed internally) |
| V | `[B, NKH, Sk, DH]` | Value tensor |
| attn_mask | `[B, NQH/1, Sq, Sk]` | Optional attention mask |
| page_table | `[B, max_blocks]` | Page table for chunked mode |
| attention_sink | `[1, NQH, 1, 1]` | Optional attention sink values |

### Key Dimensions

```cpp
B         = batch size
NQH       = number of query heads
NKH       = number of key/value heads (NQH >= NKH, NQH % NKH == 0)
Sq        = query sequence length
Sk        = key/value sequence length
DH        = head dimension
vDHt      = value head dimension in tiles (for MLA)
```

### Chunking Parameters

The operation processes Q and K/V in chunks to fit in L1 SRAM:

```cpp
Sq_chunk_t = q_chunk_size / TILE_HEIGHT  // Q chunk in tiles
Sk_chunk_t = k_chunk_size / TILE_HEIGHT  // K chunk in tiles
q_num_chunks = padded_Sq / q_chunk_size
k_num_chunks = padded_Sk / k_chunk_size
```

---

## Circular Buffer Configuration

### Input Circular Buffers

| CB Index | Name | Size (tiles) | Purpose | Buffering |
|----------|------|--------------|---------|-----------|
| c_0 | cb_q_in | `Sq_chunk_t * DHt * q_buffer_factor` | Q input staging | Single/Double |
| c_1 | cb_k_in | `Sk_chunk_t * DHt * 2` | K input (transposed) | Double |
| c_2 | cb_v_in | `Sk_chunk_t * vDHt * 2` | V input | Double |
| c_3 | cb_mask_in | `Sq_chunk_t * Sk_chunk_t * 2` | Attention mask | Double |
| c_4 | cb_attention_sink | `Sq_chunk_t` | Attention sink values | Single |
| c_5 | cb_identity_scale | 1 | Identity scalar (1.0) | Single |
| c_6 | cb_page_table | `page_table_stick_size` | Page table (chunked mode) | Single |
| c_7 | cb_col_identity | 1 | Column identity for reduce | Single |

### Intermediate Circular Buffers

| CB Index | Name | Size (tiles) | Purpose |
|----------|------|--------------|---------|
| c_24 | cb_qk_im | `Sq_chunk_t * Sk_chunk_t` | Q @ K^T intermediate |
| c_25 | cb_out_im_A | `Sq_chunk_t * vDHt` | Output intermediate (ping) |
| c_26 | cb_out_im_B | `Sq_chunk_t * vDHt` | Output intermediate (pong) |
| c_27 | cb_max_A | `Sq_chunk_t` | Current max (ping) |
| c_28 | cb_max_B | `Sq_chunk_t` | Previous max (pong) |
| c_29 | cb_sum_A | `Sq_chunk_t` | Current sum (ping) |
| c_30 | cb_sum_B | `Sq_chunk_t` | Previous sum (pong) |
| c_31 | cb_exp_max_diff | `Sq_chunk_t` | exp(prev_max - cur_max) |

### Output Circular Buffer

| CB Index | Name | Size (tiles) | Purpose |
|----------|------|--------------|---------|
| c_16 | cb_out | `Sq_chunk_t * vDHt` | Final output |

---

## Compute Kernel Deep Dive

The compute kernel (`sdpa.cpp`) implements the FlashAttention-style tiled computation. This section provides an exhaustive analysis of each phase.

### High-Level Loop Structure

```cpp
for (phase : num_phases)
  for (batch : local_batch_start..local_batch_end)
    for (head : local_nh_start..local_nh_end)
      for (q_iter : q_chunks_per_core)
        // Process one Q chunk against all relevant K chunks
        for (k_chunk : 0..q_high_idx/Sk_chunk_t)
          // Core SDPA computation phases
```

### Phase 1: Q @ K^T Matrix Multiplication

**Purpose:** Compute attention scores between Q and K chunks.

**Code Location:** Lines 144-159 in `sdpa.cpp`

```cpp
pack_reconfig_data_format(cb_qk_im);
matmul_blocks(
    cb_q_in,           // Q chunk [Sq_chunk_t, DHt]
    cb_k_in,           // K chunk [Sk_chunk_t, DHt] (transposed)
    cb_qk_im,          // Output [Sq_chunk_t, Sk_chunk_t]
    Sq_chunk_t,        // M dimension
    Sk_chunk_t,        // N dimension
    DHt,               // K dimension
    qk_num_blocks,
    qk_in0_num_subblocks,
    qk_in1_num_subblocks,
    qk_in0_block_w,
    qk_subblock_h,
    qk_subblock_w,
    true /*transpose*/ // K is transposed
);
```

**matmul_blocks Implementation Details:**

The `matmul_blocks` function in `compute_common.hpp` (lines 621-692) implements blocked matrix multiplication:

1. **Initialization:** `mm_block_init_short()` configures the math unit for matmul
2. **Blocking Strategy:** Outer loop over `in0_num_subblocks`, inner loop over `in1_num_subblocks`
3. **Tile Acquisition:** Uses `tile_regs_acquire()` / `tile_regs_release()` for DST register management
4. **Core Operation:** `matmul_block()` performs the actual FMA operations
5. **Packing:** `pack_tile<true>()` writes results with offset addressing

**Subblock Configuration:**
```cpp
dst_size = fp32_dest_acc_en ? 4 : 8;  // DST register capacity
qk_out_subblock_w = min(Sk_chunk_t, dst_size);
qk_out_subblock_h = (qk_out_subblock_w == Sk_chunk_t) ? min(Sq_chunk_t, dst_size/qk_out_subblock_w) : 1;
```

### Phase 2: Mask Application (Conditional)

**Purpose:** Apply causal mask, sliding window mask, or user-provided mask.

**Code Location:** Lines 173-192 in `sdpa.cpp`

```cpp
if constexpr (is_causal || sliding_window_size > 0) {
    if (!(q_low_idx >= k_high_idx) || sliding_window_size > 0) {
        reconfig_data_format(cb_qk_im, cb_mask_in);
        add_block_inplace(cb_qk_im, cb_mask_in, qk_chunk_tiles);
    }
} else if constexpr (use_provided_mask) {
    reconfig_data_format(cb_qk_im, cb_mask_in);
    add_block_inplace(cb_qk_im, cb_mask_in, qk_chunk_tiles);
} else if constexpr (use_padded_mask) {
    if (k_chunk == k_num_chunks - 1) {
        add_block_inplace(cb_qk_im, cb_mask_in, qk_chunk_tiles);
    }
}
```

**add_block_inplace Implementation:**

```cpp
void add_block_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t num_tiles) {
    add_tiles_init(in0_cb, in1_cb);
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst();
        add_tiles(in0_cb, in1_cb, i, i, 0);
        pack_tile(0, in0_cb);  // Write back to in0_cb
        release_dst();
    }
    cb_pop_front(in1_cb, num_tiles);
    cb_pop_front(in0_cb, num_tiles);
    cb_reserve_back(in0_cb, num_tiles);
    cb_push_back(in0_cb, num_tiles);
}
```

**Mask Types Generated by Writer:**
- **Causal mask:** Zero below diagonal, -inf above
- **Sliding window mask:** Combines leading and trailing diagonal offsets
- **Padded mask:** Vertical -inf for padding columns
- **BFP4 format:** Uses 0xC mantissa for -inf representation

### Phase 3: Row-wise Max Reduction

**Purpose:** Compute running maximum for numerical stability (softmax denominator normalization).

**Code Location:** Lines 201-208 in `sdpa.cpp`

```cpp
reconfig_data_format(cb_qk_im, cb_identity_scale_in);
reduce_c<
    PoolType::MAX,
    ReduceDim::REDUCE_ROW,
    cb_qk_im,
    cb_identity_scale_in,
    Sq_chunk_t,
    Sk_chunk_t
>(alias_cur_max, alias_prev_max, k_chunk > 0);
```

**reduce_c Implementation Details (lines 56-118):**

1. **Granularity:** Processes `REDUCE_GRANULARITY` rows at a time (power of 2, based on DST capacity)
2. **Optional Eltwise Max:** If `do_eltwise_max=true`, combines with previous max values
3. **Special Copy:** Uses `sdpa_reduce_copy_tile_to_dst_init_short()` for face-transposed DST layout
4. **Reduce Operation:** `reduce_block_max_row<cols>()` performs row-wise reduction

```cpp
// Key reduce loop
for (uint32_t g = 0; g < granularity; g++) {
    cb_wait_front(in0_cb, in0_wait_tiles);
    acquire_dst();

    if (do_eltwise_max) {
        // Copy previous max into DST with transposed faces
        sdpa_reduce_copy_tile_to_dst_init_short(prev_cb);
        for (uint32_t i = 0; i < dst_tiles; i++) {
            copy_tile(prev_cb, (row_start_idx + i), i);
        }
    }

    // Row-wise max reduction
    reduce_block_max_row_init<cols>();
    for (uint32_t i = 0; i < dst_tiles; i++) {
        reduce_block_max_row<cols>(in0_cb, scale_cb, (row_start_idx + i) * cols, i);
    }
    reduce_block_max_row_uninit();

    // Pack results
    for (uint32_t i = 0; i < dst_tiles; i++) {
        pack_tile<true>(i, out_cb, (row_start_idx + i));
    }
    release_dst();
}
```

### Phase 4: Fused Subtract-Exp-Reduce (sub_exp_block_bcast_cols_inplace)

**Purpose:** Compute `exp((QK - max) * scale)` and partial row sum simultaneously.

**Code Location:** Lines 219-220 in `sdpa.cpp`

```cpp
sub_exp_block_bcast_cols_inplace<cb_qk_im, Sq_chunk_t, Sk_chunk_t, scale_fp32, true>(
    alias_cur_max, alias_cur_sum);
```

**This is the most heavily optimized function.** Implementation (lines 193-253 in compute_common.hpp):

```cpp
template <uint32_t in0_cb, uint32_t rows, uint32_t cols, uint32_t scale_fp32, bool write_result_inplace = true>
void sub_exp_block_bcast_cols_inplace(uint32_t in1_cb, uint32_t reduce_cb) {
    sub_bcast_cols_init_short(in0_cb, in1_cb);
    exp_tile_init<true, true, scale_fp32>();  // Scale fused into exp

    cb_wait_front(in0_cb, rows * cols);
    cb_wait_front(in1_cb, rows);
    cb_reserve_back(reduce_cb, rows);

    if constexpr (write_result_inplace) {
        cb_pop_front(in0_cb, rows * cols);
        cb_reserve_back(in0_cb, rows * cols);
    }

    // Process with SUB_EXP_GRANULARITY tiles at a time
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t u = 0; u < granularity; u++) {
            tile_regs_acquire();
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                sub_tiles_bcast_cols(in0_cb, in1_cb, in0_index, i, j);
                exp_tile<true, true>(j);  // exp with scaling
                in0_index++;
            }
            tile_regs_commit();
            tile_regs_wait();

            // Write exp results back (if inplace)
            if constexpr (write_result_inplace) {
                for (uint32_t j = 0; j < dst_tiles; ++j) {
                    pack_tile(j, in0_cb);
                }
            }

            // L1 Accumulation for partial row sum
            if (u > 0) {
                PACK((llk_pack_reconfig_l1_acc(1)));  // Enable L1 accumulation
            }
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                pack_tile<true>(j, reduce_cb, i);  // Pack to same location
                if (u == 0 && j == 0) {
                    PACK((llk_pack_reconfig_l1_acc(1)));  // Start accumulating
                }
            }
            tile_regs_release();
            PACK((llk_pack_reconfig_l1_acc(0)));  // Disable L1 accumulation
        }

        if constexpr (write_result_inplace) {
            cb_push_back(in0_cb, cols);  // Granular push for early unpack
        }
    }
    cb_push_back(reduce_cb, rows);
}
```

**Key Optimizations:**
1. **Scale Fused into Exp:** `exp((x - max) * scale)` computed in single pass
2. **L1 Accumulation:** Uses hardware L1 accumulation for partial row sum
3. **Granular CB Push:** Enables compute/dataflow overlap

### Phase 5: Attention @ V Matrix Multiplication

**Purpose:** Compute weighted sum of values.

**Code Location:** Lines 222-236 in `sdpa.cpp`

```cpp
matmul_blocks(
    cb_qk_im,          // Attention weights [Sq_chunk_t, Sk_chunk_t]
    cb_v_in,           // V chunk [Sk_chunk_t, vDHt]
    alias_mm2_cur_out, // Output [Sq_chunk_t, vDHt]
    Sq_chunk_t,
    vDHt,
    Sk_chunk_t,
    out_num_blocks,
    out_in0_num_subblocks,
    out_in1_num_subblocks,
    out_in0_block_w,
    out_subblock_h,
    out_subblock_w,
    false /*transpose*/ // V is not transposed
);
```

### Phase 6: Running Statistics Update (k_chunk > 0)

**Purpose:** Rescale previous statistics when max changes between K chunks.

**Code Location:** Lines 242-266 in `sdpa.cpp`

```cpp
if (k_chunk > 0) {
    // 6a: Compute exp(prev_max - cur_max) * scale
    sub_exp_block<scale_fp32>(alias_prev_max, alias_cur_max, cb_exp_max_diff, Sq_chunk_t);
    cb_pop_front(alias_prev_max, Sq_chunk_t);

    // 6b: Rescale previous sum
    mul_tiles_bcast_cols_inplace(alias_prev_sum, cb_exp_max_diff, Sq_chunk_t);

    // 6c: Add to current sum
    add_block_inplace(alias_cur_sum, alias_prev_sum, Sq_chunk_t);

    // 6d: Rescale and accumulate previous output
    mul_block_bcast_cols<Sq_chunk_t, vDHt>(
        alias_mm2_prev_out, cb_exp_max_diff, alias_mm2_cur_out, true /*pack_accumulate*/);
}
```

**sub_exp_block Implementation (lines 456-484):**

```cpp
template <uint32_t scale_fp32>
void sub_exp_block(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t num_tiles) {
    sub_tiles_init(in0_cb, in1_cb);
    exp_tile_init<EXP_APPROX_MODE, false>();

    constexpr uint16_t scale_bf16 = scale_fp32 >> 16;  // Extract bf16 from fp32

    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst();
        sub_tiles(in0_cb, in1_cb, i, i, 0);
        MATH((exp_tile_first_column<EXP_APPROX_MODE>(0, scale_bf16)));  // Custom VectorMode::C exp
        pack_tile(0, out_cb);
        cb_push_back(out_cb, 1);
        release_dst();
    }
}
```

**exp_tile_first_column Optimization:**

Only computes exp for the first column of each face (columns 0-7), since max values are stored in column format:

```cpp
template <bool SDPA_EXP_APPROX_MODE>
void calculate_exponential_first_column(int scale_bf16) {
    constexpr int ITERATIONS_HALF_FACE = 4;  // Only 4 iterations for half face
    for (int d = 0; d < ITERATIONS_HALF_FACE; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::vFloat result = _calculate_exponential_piecewise_<...>(val, scale_bf16);
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg += 2;  // Stride by 2 to skip columns 8-15
    }
}
```

### Phase 7: Ping-Pong Buffer Swap

**Purpose:** Prepare for next K chunk iteration.

**Code Location:** Lines 268-271 in `sdpa.cpp`

```cpp
std::swap(alias_prev_sum, alias_cur_sum);
std::swap(alias_mm2_prev_out, alias_mm2_cur_out);
std::swap(alias_prev_max, alias_cur_max);
```

### Phase 8: Final Row Reduction

**Purpose:** Complete the partial row sum to get full softmax denominator.

**Code Location:** Line 276 in `sdpa.cpp`

```cpp
matmul_reduce<Sq_chunk_t>(cb_col_identity, alias_prev_sum);
```

**matmul_reduce Implementation (lines 694-741):**

Uses matmul with a column of ones to sum across rows:

```cpp
template <uint32_t M>
void matmul_reduce(uint32_t in1_cb, const uint32_t& out_cb) {
    constexpr uint32_t N = 1;
    constexpr uint32_t subblock_h = STATS_GRANULARITY;

    mm_block_init_short(out_cb, in1_cb, 0, 1, subblock_h, 1);

    cb_wait_front(in1_cb, N);
    cb_wait_front(out_cb, M);

    for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; ++in0_subblock) {
        tile_regs_acquire();
        matmul_block(out_cb, in1_cb, 0, 0, 0, 0, 1, subblock_h, 1);
        tile_regs_commit();

        cb_pop_front(out_cb, subblock_h);
        tile_regs_wait();
        for (uint32_t i = 0; i < subblock_h; i++) {
            pack_tile(i, out_cb);
        }
        tile_regs_release();
        cb_push_back(out_cb, subblock_h);
    }
}
```

### Phase 9: Attention Sink Processing (Optional)

**Purpose:** Include attention sink in softmax denominator without contributing to output.

**Code Location:** Lines 289-329 in `sdpa.cpp`

```cpp
if constexpr (use_attention_sink) {
    // Update max with sink values
    reduce_c<PoolType::MAX, ...>(alias_cur_max, alias_prev_max, true);

    // Rescale previous statistics
    sub_exp_block<scale_fp32>(alias_prev_max, alias_cur_max, cb_exp_max_diff, Sq_chunk_t);
    mul_tiles_bcast_cols_inplace(alias_prev_sum, cb_exp_max_diff, Sq_chunk_t);

    // Add sink contribution to sum
    sub_exp_block_bcast_cols_inplace<cb_attention_sink, Sq_chunk_t, 1, scale_fp32, false>(
        alias_cur_max, alias_cur_sum);
    add_block_inplace(alias_cur_sum, alias_prev_sum, Sq_chunk_t);

    // Swap statistics
    std::swap(alias_prev_sum, alias_cur_sum);
    std::swap(alias_prev_max, alias_cur_max);

    // Rescale output (no new V contribution)
    mul_block_bcast_cols<Sq_chunk_t, vDHt>(
        alias_mm2_prev_out, cb_exp_max_diff, alias_mm2_cur_out, false);
    std::swap(alias_mm2_prev_out, alias_mm2_cur_out);
}
```

### Phase 10: Final Normalization

**Purpose:** Divide output by softmax sum.

**Code Location:** Lines 330-335 in `sdpa.cpp`

```cpp
// Compute reciprocal of sum
recip_block_inplace(alias_prev_sum, Sq_chunk_t);

// Multiply output by 1/sum
pack_reconfig_data_format(cb_out);
mul_block_bcast_cols<Sq_chunk_t, vDHt>(alias_mm2_prev_out, alias_prev_sum, cb_out, false);
```

**recip_block_inplace Optimization:**

Uses custom `recip_tile_first_column` that only computes reciprocal for the first column:

```cpp
template <bool legacy_compat = true>
void calculate_recip_first_column() {
    constexpr int ITERATIONS_HALF_FACE = 4;
    for (int d = 0; d < ITERATIONS_HALF_FACE; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat out = _reciprocal_compat_<APPROX ? 2 : 3>(in);
        sfpi::dst_reg[0] = out;
        sfpi::dst_reg += 2;  // Skip columns 8-15
    }
}
```

---

## Reader Kernel Analysis

### Data Fetching Strategy

The reader kernel (`reader_interleaved.cpp`) fetches Q, K, V, and mask data from DRAM into L1 circular buffers.

### Key Functions

**1. Q Chunk Reading:**
```cpp
read_chunk_with_padding<q_tile_bytes>(
    q_reader, cb_q_in, q_tile_id,
    q_row_tile_count, DHt,  // src dimensions
    Sq_chunk_t, DHt,        // dst dimensions
    barrier_threshold);
```

**2. K Chunk Reading (with transpose):**
```cpp
read_chunk_with_padding<k_tile_bytes>(
    k_reader, cb_k_in, k_start_tile_id,
    k_row_tile_count, DHt,
    Sk_chunk_t, DHt,
    barrier_threshold,
    true  // transpose=true for K
);
```

**3. Paged K/V Reading (chunked mode):**
```cpp
read_paged_chunk_with_padding<NKH, block_size_t, DHt>(
    k_reader, cb_k_in, kv_head,
    k_chunk_start_row_num, k_row_tile_count,
    DHt, Sk_chunk_t, DHt,
    k_tile_bytes, barrier_threshold,
    page_table_ptr, true  // transpose for K
);
```

### TensorAccessor Usage

Uses compile-time tensor accessor arguments for efficient address calculation:

```cpp
constexpr auto q_args = TensorAccessorArgs<22>();
constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
// ...
const auto q_reader = TensorAccessor(q_args, q_addr, q_tile_bytes);
```

### Padding Handling

The `read_chunk_with_padding` template handles cases where valid sequence length does not align with chunk boundaries:

```cpp
// Zero out padding tiles
for (uint32_t row = 0; row < dst_rows; ++row) {
    for (uint32_t col = 0; col < dst_cols; ++col) {
        if (row < src_rows && col < src_cols) continue;
        uint32_t tile_id = transpose ? col * dst_rows + row : row * dst_cols + col;
        fill_tile_zeros<tile_bytes>(cb_id, tile_id);
    }
}
```

### Attention Sink Reading

Reads a single tile per head and replicates to all Q positions:

```cpp
if constexpr (use_attention_sink) {
    const uint32_t sink_tile_id = attention_sink_tile_shape.id_of(0, nq, 0, 0);
    noc_async_read_tile(sink_tile_id, attention_sink_reader, attention_sink_write_ptr);
    noc_async_read_barrier();

    fill_attention_sink_tiles<attention_sink_tile_bytes>(
        cb_attention_sink, Sq_chunk_t, attention_sink_write_ptr);
}
```

---

## Writer Kernel Analysis

### Responsibilities

The writer kernel handles:
1. Generating causal/sliding window masks
2. Generating scalar constants (identity, scale)
3. Writing output to DRAM

### Mask Generation

**Causal Mask Generation:**
```cpp
generate_mask<cb_mask_in>(
    Sq_chunk_t, Sk_chunk_t, offset_q_chunk, k_chunk,
    is_causal, sliding_window_size);
```

**MaskType Classification:**
- `FULLY_ALLOWED`: Zero tile (no masking)
- `FULLY_MASKED`: -inf tile
- `PARTIAL_MASK`: Diagonal tile with custom offsets

**BFP4 Mask Format:**

The mask uses BFP4_B format (4-bit mantissa, shared exponent):
- -inf: exponent=0xFF, mantissa=0xC
- zero: exponent=0x00, mantissa=0x0

```cpp
constexpr uint32_t NEG_INF_EXP = 0xFFFFFFFF;
constexpr uint32_t NEG_INF_MANT = 0xCCCCCCCC;  // 8 x 0xC per uint32
```

### Scalar Generation

```cpp
dataflow_kernel_lib::generate_reduce_scaler(cb_identity_scale_in, identity_scalar_packed);
generate_bcast_col_scalar(cb_col_identity, identity_scalar_packed);
```

### Output Writing

```cpp
for (uint32_t row = 0; row < out_row_tile_count; ++row) {
    for (uint32_t col = 0; col < vDHt; ++col) {
        noc_async_write_tile(out_tile_id, out_writer, l1_read_addr);
        ++out_tile_id;
        l1_read_addr += tile_bytes;

        if (++barrier_count == barrier_threshold) {
            noc_async_writes_flushed();
            barrier_count = 0;
        }
    }
}
```

---

## Low-Level Optimizations

### 1. Scale Fusion into Exponential

Instead of computing `exp(x - max) * scale`, the kernel computes `exp((x - max) * scale)` by fusing scale into the exp operation:

```cpp
exp_tile_init<true, true, scale_fp32>();  // Scale passed as template parameter
exp_tile<true, true>(j);
```

This saves one multiply per element.

### 2. L1 Accumulation for Partial Sums

Uses hardware L1 accumulation to avoid extra CB push/pop:

```cpp
PACK((llk_pack_reconfig_l1_acc(1)));  // Enable
pack_tile<true>(j, reduce_cb, i);      // Accumulates to same address
PACK((llk_pack_reconfig_l1_acc(0)));  // Disable
```

### 3. VectorMode::C Optimizations

Many operations only need to process the first column (statistics are column vectors):

- `exp_tile_first_column()`: Only processes columns 0-7 of each face
- `recip_tile_first_column()`: Same optimization for reciprocal
- Saves 50% of SFPU cycles for these operations

### 4. BFP4 Mask Format

Using BFP4_B (4-bit mantissa) for masks reduces memory bandwidth by 4x compared to BF16:
- Tile size: 576 bytes (BFP4) vs 2048 bytes (BF16)
- Sufficient precision for -inf/0 mask values

### 5. Granular CB Operations

Granular push/pop enables compute-dataflow overlap:

```cpp
for (uint32_t i = 0; i < rows; ++i) {
    // ... compute row ...
    cb_push_back(in0_cb, cols);  // Push after each row
}
```

### 6. Balanced Q Parallel

For causal attention, distributes Q chunks to balance work:

```cpp
#if defined BALANCED_Q_PARALLEL
if (q_iter < q_chunk_div_2) {
    q_chunk = local_q_start + q_iter;  // Front chunks
} else {
    q_chunk = q_num_chunks - 1 - (local_q_start + back_q_iter);  // Back chunks
}
#endif
```

### 7. Double Buffering

K and V use double-buffered CBs for compute/dataflow overlap:
```cpp
k_tiles = Sk_chunk_t * DHt * 2;  // Double buffer
v_tiles = Sk_chunk_t * vDHt * 2;
```

### 8. Barrier Thresholding

Amortizes NoC barrier overhead:
```cpp
constexpr uint32_t barrier_threshold = get_barrier_read_threshold<tile_bytes, num_cores>();
// Barrier only every N tiles
if (++barrier_count == barrier_threshold) {
    noc_async_read_barrier();
    barrier_count = 0;
}
```

---

## Kernel Capabilities and Limitations

### Capabilities

| Feature | Support | Notes |
|---------|---------|-------|
| Causal masking | Yes | Generated on-device |
| Sliding window | Yes | Combined with causal |
| User mask | Yes | Must divide chunk size |
| GQA (grouped query attention) | Yes | NQH >= NKH, NQH % NKH == 0 |
| MLA (multi-latent) | Yes | Shared K/V buffer |
| Paged attention | Yes | Chunked mode with page table |
| Attention sink | Yes | Virtual K chunk |
| Sequence padding | Yes | Automatic zero-fill |
| Data types | BF16, BFP8, BFP4 | For inputs and masks |
| FP32 accumulation | Yes | Via compute_kernel_config |
| Approximate exp | Yes | Via EXP_APPROX_MODE |

### Limitations

| Limitation | Description | Workaround |
|------------|-------------|------------|
| Interleaved only | No sharded tensor support | Must use DRAM/L1 interleaved |
| Tile alignment | Dimensions must be multiples of 32 | Padding handled internally |
| Fixed head dim | DH and vDH fixed per call | MLA allows different vDH |
| Causal + mask | Cannot combine causal with user mask | Use user mask for full control |
| Sequence length | Memory-limited by L1 size | Use chunking for long sequences |
| Batch parallelism | Limited by core count | Use multi-device for large batch |

### Memory Constraints

L1 usage per core (typical configuration):
```
Q buffer:     Sq_chunk_t * DHt * tile_size
K buffer:     2 * Sk_chunk_t * DHt * tile_size
V buffer:     2 * Sk_chunk_t * vDHt * tile_size
Mask buffer:  2 * Sq_chunk_t * Sk_chunk_t * mask_tile_size
QK buffer:    Sq_chunk_t * Sk_chunk_t * tile_size
Out buffers:  2 * Sq_chunk_t * vDHt * tile_size
Stats:        8 * Sq_chunk_t * stats_tile_size
Output:       Sq_chunk_t * vDHt * tile_size
```

---

## Performance Considerations

### Optimal Chunk Sizes

The program factory computes optimal subblock configurations:

```cpp
dst_size = fp32_dest_acc_en ? 4 : 8;  // DST capacity
qk_out_subblock_w = min(Sk_chunk_t, dst_size);
qk_out_subblock_h = (qk_out_subblock_w == Sk_chunk_t) ?
                    min(Sq_chunk_t, dst_size/qk_out_subblock_w) : 1;

// Optimization: prefer 2x4 subblock for better matmul utilization
if (qk_out_subblock_w == dst_size && qk_out_subblock_h == 1 &&
    Sk_chunk_t % 2 == 0 && Sq_chunk_t % 2 == 0) {
    qk_out_subblock_w /= 2;
    qk_out_subblock_h = 2;
}
```

### Parallelization Strategy

Work distribution priority:
1. Batch dimension (highest priority)
2. Num heads dimension
3. Q chunks dimension (lowest priority)

```cpp
batch_parallel_factor = min(B, num_cores);
nh_parallel_factor = min(num_cores / batch_parallel_factor, NQH);
q_parallel_factor = min(num_cores / (batch_parallel_factor * nh_parallel_factor), q_num_chunks);
```

### Compute vs Memory Bound

- **Compute bound:** Large head dimensions (DHt), small batch
- **Memory bound:** Small head dimensions, large sequences
- **Balanced:** Typical transformer configurations

### Recommended Configurations

| Use Case | q_chunk_size | k_chunk_size | Notes |
|----------|--------------|--------------|-------|
| Short sequences | 32 | 32 | Default, minimal L1 |
| Long sequences | 64-128 | 64-128 | Better compute efficiency |
| Large head dim | 32 | 32 | Memory constrained |
| Decode (Sk >> Sq) | 32 | 256 | Asymmetric chunking |

---

## Summary

The SDPA kernel implements a highly optimized FlashAttention-style computation with:

1. **Tiled Processing:** Q and K/V processed in chunks to fit L1
2. **Online Softmax:** Running max/sum statistics for numerical stability
3. **Fused Operations:** Scale into exp, L1 accumulation for sums
4. **Memory Efficiency:** BFP4 masks, double buffering, granular CB ops
5. **Flexibility:** Supports causal, masked, MLA, paged, and sliding window attention

The implementation balances compute efficiency with memory constraints through careful subblock sizing and parallelization strategies.
