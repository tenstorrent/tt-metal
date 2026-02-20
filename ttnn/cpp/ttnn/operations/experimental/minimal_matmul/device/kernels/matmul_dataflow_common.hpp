// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <algorithm>
#include <tuple>
#include <utility>
#include "api/dataflow/dataflow_api.h"

namespace detail {
/**
 * Helper to create tuple of TensorAccessors from TensorAccessorArgs tuple.
 * Unlike make_tensor_accessor_tuple, this takes the actual page_size value,
 * not a Compile Time Argument (CTA) index.
 */
template <typename... Args, uint32_t... Indexes>
auto make_tensor_accessor_tuple_with_page_size(
    const std::tuple<Args...>& args_tuple,
    uint32_t address_rt_arg_index_start,
    uint32_t page_size,
    std::integer_sequence<uint32_t, Indexes...>) {
    return std::make_tuple(TensorAccessor(
        std::get<Indexes>(args_tuple), get_arg_val<uint32_t>(address_rt_arg_index_start + Indexes), page_size)...);
}
}  // namespace detail

/**
 * Create a tuple of TensorAccessors from a tuple of TensorAccessorArgs.
 * Each tensor gets its address from consecutive RT args starting at address_rt_arg_index_start.
 * All tensors share the same page_size (actual value, not CTA index).
 */
template <typename... Args>
auto make_tensor_accessor_tuple_uniform_page_size(
    const std::tuple<Args...>& args_tuple, uint32_t address_rt_arg_index_start, uint32_t page_size) {
    return detail::make_tensor_accessor_tuple_with_page_size(
        args_tuple, address_rt_arg_index_start, page_size, std::make_integer_sequence<uint32_t, sizeof...(Args)>());
}
void fill_zeros_async(uint32_t write_addr, uint32_t tile_bytes) {
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_addr);
    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    // Fill tile with zeros
    uint32_t bytes_left = tile_bytes;
    for (;;) {
        uint32_t read_size = bytes_left > MEM_ZEROS_SIZE ? MEM_ZEROS_SIZE : bytes_left;
        noc_async_read(zeros_noc_addr, write_addr, read_size);
        write_addr += read_size;
        bytes_left -= read_size;
        if (bytes_left == 0) {
            break;
        }
    }
}

struct TensorShape2D {
    uint32_t logical_d0;
    uint32_t logical_d1;
    uint32_t padded_d0;
    uint32_t padded_d1;
    // Constructor to initialize with 2D shape
    TensorShape2D(uint32_t _d0, uint32_t _d1, uint32_t _padded_d0, uint32_t _padded_d1) :
        logical_d0(_d0), logical_d1(_d1), padded_d0(_padded_d0), padded_d1(_padded_d1) {
        ASSERT(_d0 > 0);
        ASSERT(_d1 > 0);
        ASSERT(_d0 <= _padded_d0);
        ASSERT(_d1 <= _padded_d1);
    }
};

/**
 * Read a block of in0 from a potentially padded tensor.
 * Since this is for matmul, no need to read when M >= logical_M
 * Otherwise, if K >= logical_K, fill with zeros.
 */
template <
    uint32_t M_block_tiles,
    uint32_t K_block_tiles,
    typename TensorAccessorType
#ifdef READ_FROM_LOCAL_INPUT
    ,
    typename LocalTensorAccessorType
#endif
    >
void read_in0_block_sync(
    const TensorAccessorType& tensor_accessor,
    const TensorShape2D& shape,
    uint32_t write_ptr,
    uint32_t tile_size_bytes,
#ifdef READ_FROM_LOCAL_INPUT
    const LocalTensorAccessorType& in3_accessor,
    uint32_t local_k_start,
    uint32_t local_k_end,
    uint32_t input_tensor_Wt,
#endif
    uint32_t d0_start,
    uint32_t d0_end,
    uint32_t d1_start,
    uint32_t d1_end) {
    ASSERT(d0_end > d0_start);
    ASSERT(d1_end > d1_start);

    for (uint32_t i = d0_start; i < d0_end; i++) {
        if (i >= shape.logical_d0) {
            break;
        }
        for (uint32_t j = d1_start; j < d1_end; j++) {
            if (j < shape.logical_d1) {
#ifdef READ_FROM_LOCAL_INPUT
                if (local_k_start <= j && j <= local_k_end) {
                    // read from self_tensor_accessor
                    uint32_t tile_id = i * input_tensor_Wt + (j - local_k_start);
                    noc_async_read_tile(tile_id, in3_accessor, write_ptr);
                } else {
#endif
                    uint32_t tile_id = i * shape.logical_d1 + j;
                    noc_async_read_tile(tile_id, tensor_accessor, write_ptr);
#ifdef READ_FROM_LOCAL_INPUT
                }
#endif
            } else {
                fill_zeros_async(write_ptr, tile_size_bytes);
            }
            write_ptr += tile_size_bytes;
        }
        // finish up incrementing write_ptr if (d1_end - d1_start) < K_block_tiles
        write_ptr += (K_block_tiles - (d1_end - d1_start)) * tile_size_bytes;
    }
    noc_async_read_barrier();
}

/**
 * Read a block of in1 from a potentially padded tensor.
 * Since this is for matmul, no need to read when N >= logical_N
 * Otherwise, if K >= logical_K, fill with zeros.
 */
template <uint32_t K_block_tiles, uint32_t N_block_tiles, typename TensorAccessorType>
void read_in1_block_sync(
    const TensorAccessorType& tensor_accessor,
    const TensorShape2D& shape,
    uint32_t write_ptr,
    uint32_t tile_size_bytes,
    uint32_t d0_start,
    uint32_t d0_end,
    uint32_t d1_start,
    uint32_t d1_end) {
    ASSERT(d0_end > d0_start);
    ASSERT(d1_end > d1_start);
    for (uint32_t i = d0_start; i < d0_end; i++) {
        for (uint32_t j = d1_start; j < d1_end; j++) {
            if (j >= shape.logical_d1) {
                write_ptr += tile_size_bytes;
                continue;
            }
            if (i < shape.logical_d0) {
                uint32_t tile_id = i * shape.logical_d1 + j;
                noc_async_read_tile(tile_id, tensor_accessor, write_ptr);
            } else {
                fill_zeros_async(write_ptr, tile_size_bytes);
            }
            write_ptr += tile_size_bytes;
        }
        // finish up incrementing write_ptr if (d1_end - d1_start) < K_block_tiles
        write_ptr += (N_block_tiles - (d1_end - d1_start)) * tile_size_bytes;
    }
    noc_async_read_barrier();
}

/**
 * Write a block of output to a potentially padded tensor.
 * Skip writing when M >= logical_M or N >= logical_N
 */
template <uint32_t M_block_tiles, uint32_t N_block_tiles, typename TensorAccessorType>
void write_block_sync(
    const TensorAccessorType& tensor_accessor,
    const TensorShape2D& shape,
    uint32_t read_ptr,
    uint32_t tile_size_bytes,
    uint32_t d0_start,
    uint32_t d0_end,
    uint32_t d1_start,
    uint32_t d1_end) {
    ASSERT(d0_end > d0_start);
    ASSERT(d1_end > d1_start);

    for (uint32_t i = d0_start; i < d0_end; i++) {
        if (i >= shape.logical_d0) {
            break;
        }
        for (uint32_t j = d1_start; j < d1_end; j++) {
            if (j >= shape.logical_d1) {
                read_ptr += tile_size_bytes;
                continue;
            }
            uint32_t tile_id = i * shape.logical_d1 + j;
            noc_async_write_tile(tile_id, tensor_accessor, read_ptr);
            read_ptr += tile_size_bytes;
        }
        // finish up incrementing read_ptr if (d1_end - d1_start) < N_block_tiles
        read_ptr += (N_block_tiles - (d1_end - d1_start)) * tile_size_bytes;
    }
    noc_async_writes_flushed();
}

/**
 * Read ternary inputs (ternary_a and ternary_b) and write data to CB
 *
 * For ternary_a: read M_block_tiles * N_block_tiles tiles (full block)
 * For ternary_b: only read 1 row of tiles (N_block_tiles). Compute kernel will bcast this row.
 *
 * Performance optimization: Unlike read_in0_block_sync and read_in1_block_sync, pushes ternary_a
 * tiles one row at a time. This allows the compute kernel to begin processing addcmul operations
 * as soon as the first row is ready, rather than waiting for the entire block. This overlapping
 * of data movement and compute improves overall throughput.
 */
template <uint32_t M_block_tiles, uint32_t N_block_tiles, typename TensorAccessorType>
void read_ternary_blocks_sync(
    const TensorAccessorType& ternary_a_accessor,
    const TensorAccessorType& ternary_b_accessor,
    const TensorShape2D& shape,
    uint32_t ternary_a_cb,
    uint32_t ternary_b_cb,
    uint32_t tile_size_bytes,
    uint32_t d0_start,
    uint32_t d0_end,
    uint32_t d1_start,
    uint32_t d1_end) {
    ASSERT(d0_end > d0_start);
    ASSERT(d1_end > d1_start);

    cb_reserve_back(ternary_b_cb, N_block_tiles);
    uint32_t ternary_b_write_ptr = get_write_ptr(ternary_b_cb);
    for (uint32_t n_tile_id = d1_start; n_tile_id < d1_end; n_tile_id++) {
        if (n_tile_id >= shape.logical_d1) {
            // Do not move tile data into CB if tile is outside ternary/output tensor.
            // This can happen when ternary/output tensor shape is not a multiple of block sizes:
            // For instance, if tensor shape is (M_tiles=7, N_tiles=3), but block sizes are (M_block_tiles=4,
            // N_block_tiles=4)
            break;
        }

        noc_async_read_tile(n_tile_id, ternary_b_accessor, ternary_b_write_ptr);
        ternary_b_write_ptr += tile_size_bytes;
    }
    noc_async_read_barrier();

    cb_push_back(ternary_b_cb, N_block_tiles);

    uint32_t m_id = 0;
    uint32_t i = d0_start;
    for (; i < d0_end; i++, m_id++) {
        cb_reserve_back(ternary_a_cb, N_block_tiles);

        uint32_t ternary_a_write_ptr = get_write_ptr(ternary_a_cb);
        for (uint32_t j = d1_start; j < d1_end; j++) {
            if (j >= shape.logical_d1) {
                // Do not move tile data into CB if tile is outside ternary/output tensor.
                // This can happen when ternary/output tensor shape is not a multiple of block sizes:
                // For instance, if tensor shape is (M_tiles=7, N_tiles=3), but block sizes are (M_block_tiles=4,
                // N_block_tiles=4)
                break;
            }
            if (i < shape.logical_d0) {
                uint32_t tile_id = i * shape.logical_d1 + j;
                noc_async_read_tile(tile_id, ternary_a_accessor, ternary_a_write_ptr);
            }
            ternary_a_write_ptr += tile_size_bytes;
        }
        noc_async_read_barrier();

        cb_push_back(ternary_a_cb, N_block_tiles);
    }
    for (; m_id < M_block_tiles; m_id++) {
        cb_reserve_back(ternary_a_cb, N_block_tiles);
        cb_push_back(ternary_a_cb, N_block_tiles);
    }
}

/**
 * This write method is more granular, waiting on a row of output tiles
 * in the output CB before writing those out, rather than waiting on the entire block.
 */
template <uint32_t M_block_tiles, uint32_t N_block_tiles, typename TensorAccessorType>
void write_block_sync_granular(
    const TensorAccessorType& tensor_accessor,
    const TensorShape2D& shape,
    uint32_t cb_id_out,
    uint32_t tile_size_bytes,
    uint32_t d0_start,
    uint32_t d0_end,
    uint32_t d1_start,
    uint32_t d1_end) {
    for (uint32_t m_id = 0; m_id < M_block_tiles; m_id++) {
        cb_wait_front(cb_id_out, N_block_tiles);
        uint32_t m_tile = d0_start + m_id;
        if (m_tile < d0_end && m_tile < shape.logical_d0) {
            uint32_t out_read_ptr = get_read_ptr(cb_id_out);
            for (uint32_t n_tile_id = d1_start; n_tile_id < d1_end; n_tile_id++) {
                if (n_tile_id >= shape.logical_d1) {
                    break;
                }
                uint32_t tile_id = m_tile * shape.logical_d1 + n_tile_id;
                noc_async_write_tile(tile_id, tensor_accessor, out_read_ptr);
                out_read_ptr += tile_size_bytes;
            }
        }
        cb_pop_front(cb_id_out, N_block_tiles);
    }
    noc_async_writes_flushed();
}

/**
 * Helper: dispatch to correct tuple element using fold expression.
 * Each branch calls noc_async_write_tile with the concrete TensorAccessor type.
 */
template <typename Tuple, size_t... Is>
FORCE_INLINE void write_tile_to_chunk(
    const Tuple& accessors, uint32_t chunk_idx, uint32_t tile_id, uint32_t read_ptr, std::index_sequence<Is...>) {
    // Fold expression: expands to if/else chain at compile time
    // Each branch calls noc_async_write_tile with the concrete type
    ((chunk_idx == Is ? (noc_async_write_tile(tile_id, std::get<Is>(accessors), read_ptr), void()) : void()), ...);
}

/**
 * Write a block of output to a potentially padded tensor.
 * Skip writing when M >= logical_M or N >= logical_N
 *
 * Note: Unlike write_block_sync, this function takes a tuple of accessors, rather than a single accessor.
 */
template <
    uint32_t M_block_tiles,
    uint32_t N_block_tiles,
    uint32_t N_chunks,
    uint32_t N_tiles_per_chunk,
    typename... Accessors>
void write_block_sync_split(
    const std::tuple<Accessors...>& accessors,
    const TensorShape2D& chunk_shape,
    uint32_t read_ptr,
    uint32_t tile_size_bytes,
    uint32_t d0_start,
    uint32_t d0_end,
    uint32_t d1_start,
    uint32_t d1_end) {
    ASSERT(d0_end > d0_start);
    ASSERT(d1_end > d1_start);

    const uint32_t chunk_idx_start = d1_start / N_tiles_per_chunk;
    const uint32_t tile_idx_in_chunk_start = d1_start % N_tiles_per_chunk;

    for (uint32_t i = d0_start; i < d0_end; i++) {
        // Assumes that all chunks have same number of tiles on the M-axis
        if (i >= chunk_shape.logical_d0) {
            break;
        }

        uint32_t chunk_idx = chunk_idx_start;
        uint32_t tile_idx_in_chunk = tile_idx_in_chunk_start;

        for (uint32_t j = d1_start; j < d1_end; j++, tile_idx_in_chunk++) {
            // If we've reached the end of the current chunk, move to the next one
            if (tile_idx_in_chunk >= chunk_shape.logical_d1) {
                tile_idx_in_chunk = 0;
                chunk_idx++;  // Move to next chunk; if chunk is past last one then next branch will skip padding
            }

            // Skip padding
            if (chunk_idx >= N_chunks) {
                read_ptr += tile_size_bytes;
                continue;
            }

            uint32_t tile_id_in_chunk = i * chunk_shape.logical_d1 + tile_idx_in_chunk;

            // Compile-time dispatch preserving concrete types
            write_tile_to_chunk(
                accessors, chunk_idx, tile_id_in_chunk, read_ptr, std::index_sequence_for<Accessors...>{});
            read_ptr += tile_size_bytes;
        }
        // finish up incrementing read_ptr if (d1_end - d1_start) < N_block_tiles
        read_ptr += (N_block_tiles - (d1_end - d1_start)) * tile_size_bytes;
    }
    noc_async_writes_flushed();
}

/**
 * Variadic write method for split operation with N output tensors.
 * Takes the tuple directly, preserving concrete TensorAccessor<DSpec> types for noc_async_write_tile.
 */
template <
    uint32_t M_block_tiles,
    uint32_t N_block_tiles,
    uint32_t N_chunks,
    uint32_t N_tiles_per_chunk,
    typename... Accessors>
void write_block_sync_granular_split(
    const std::tuple<Accessors...>& accessors,
    const TensorShape2D& chunk_shape,
    uint32_t cb_id_out,
    uint32_t tile_size_bytes,
    uint32_t d0_start,
    uint32_t d0_end,
    uint32_t d1_start,
    uint32_t d1_end) {
    const uint32_t chunk_idx_start = d1_start / N_tiles_per_chunk;
    const uint32_t tile_idx_in_chunk_start = d1_start % N_tiles_per_chunk;

    for (uint32_t m_id = 0; m_id < M_block_tiles; m_id++) {
        cb_wait_front(cb_id_out, N_block_tiles);
        uint32_t m_tile = d0_start + m_id;
        if (m_tile < d0_end && m_tile < chunk_shape.logical_d0) {
            uint32_t out_read_ptr = get_read_ptr(cb_id_out);

            uint32_t chunk_idx = chunk_idx_start;
            uint32_t tile_idx_in_chunk = tile_idx_in_chunk_start;

            for (uint32_t n_tile_id = d1_start; n_tile_id < d1_end; n_tile_id++, tile_idx_in_chunk++) {
                // If we've reached the end of the current chunk, move to the next one
                if (tile_idx_in_chunk >= chunk_shape.logical_d1) {
                    tile_idx_in_chunk = 0;
                    chunk_idx++;  // Move to next chunk; if chunk is past last one then next branch will skip padding
                }

                if (chunk_idx >= N_chunks) {
                    break;
                }

                uint32_t tile_id = m_tile * chunk_shape.logical_d1 + tile_idx_in_chunk;
                // Compile-time dispatch preserving concrete types
                write_tile_to_chunk(
                    accessors, chunk_idx, tile_id, out_read_ptr, std::index_sequence_for<Accessors...>{});

                out_read_ptr += tile_size_bytes;
            }
        }
        cb_pop_front(cb_id_out, N_block_tiles);
    }
    noc_async_writes_flushed();
}
