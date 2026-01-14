// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <algorithm>
#include "api/dataflow/dataflow_api.h"

#ifdef USE_MUX
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"

using namespace tt::tt_fabric::linear::experimental;
#endif

#ifdef USE_MUX
uint32_t compute_actual_k_block(
    uint32_t k_block_iter,
    uint32_t total_k_block_count,
    uint32_t my_rank,
    uint32_t k_blocks_per_device,
    uint32_t num_devices,
    bool is_forward,
    bool is_first_n_block,
    volatile tt_l1_ptr uint32_t* out_ready_semaphore,
    uint32_t& sem_target,
    bool is_injector_core,
    uint32_t& slices_received) {
#else
uint32_t compute_actual_k_block(
				uint32_t k_block_iter,
				uint32_t total_k_block_count,
				uint32_t my_rank,
				uint32_t k_blocks_per_device,
				uint32_t num_devices,
				bool is_forward) {
#endif
    // Start with self, then go backward, then forward.
    // Each time, read all k_blocks on device before moving on
    // If is_forward is false, iterate in the backwards order
    uint32_t actual_k_block_iter = is_forward ? k_block_iter : (total_k_block_count - 1 - k_block_iter);
    uint32_t device_iter = actual_k_block_iter / k_blocks_per_device;
    uint32_t device_k_block_iter = actual_k_block_iter % k_blocks_per_device;
    uint32_t direction_offset = (device_iter + 1) / 2;
    int32_t actual_device_rank = 0;
    if (device_iter % 2) {
        // Backward
        actual_device_rank = my_rank - direction_offset;
        if (actual_device_rank < 0) {
            actual_device_rank = num_devices + actual_device_rank;
        }
    } else {
        // Forward
        actual_device_rank = my_rank + direction_offset;
        if ((uint32_t)actual_device_rank >= num_devices) {
            actual_device_rank = actual_device_rank - num_devices;
        }
    }
    uint32_t k_block_start = k_blocks_per_device * actual_device_rank;
    uint32_t k_block_index = k_block_start + device_k_block_iter;
#ifdef USE_MUX
    if (device_iter > 0 && is_first_n_block) {
        // When we are not reading from local, and we are in the first forward pass through n, wait for data to arrive
        noc_semaphore_wait_min(out_ready_semaphore, sem_target + 1);
        sem_target++;
        if (device_k_block_iter == 0) {
            slices_received++;
        }
    }
#endif
    return k_block_index;
}

#ifdef USE_MUX
template <
    typename TensorAccessorType,
    typename ConnectionHandleType,
    typename ScatterPacketHdrType,
    typename UnicastPacketHdrType,
    typename SemIncPacketHdrType>
void forward_tile_to_fabric_neighbor(
    uint32_t m_tile_start,
    uint32_t k_tile_start,
    uint32_t m_block_tiles,
    uint32_t k_block_tiles,
    uint32_t num_tiles_to_write_per_packet,
    uint32_t in0_start_address,
    uint32_t output_tensor_Wt,
    const TensorAccessorType& output_addrgen,
    ConnectionHandleType mux_connection_handle,
    ScatterPacketHdrType pkt_scatter_hdr,
    UnicastPacketHdrType pkt_unicast_hdr,
    SemIncPacketHdrType pkt_hdr_sem_inc,
    uint16_t page_size,
    uint64_t out_ready_sem_noc_addr_in_pkt) {
    uint32_t tiles_to_read = m_block_tiles * k_block_tiles;
    uint32_t tiles_read = 0;
    uint32_t tile_id_start = m_tile_start * output_tensor_Wt + k_tile_start;
    uint32_t row_offset = 0;
    uint32_t pages_read_in_row = 0;
    uint32_t l1_read_addr = in0_start_address;
    while (tiles_read < tiles_to_read) {
        uint32_t tiles_remaining_to_read = tiles_to_read - tiles_read;
        uint32_t tiles_to_put_in_current_packet = std::min(tiles_remaining_to_read, num_tiles_to_write_per_packet);

        uint16_t chunk_sizes[3] = {page_size, page_size, page_size};
        uint64_t noc_addrs[4] = {0, 0, 0, 0};
        for (uint32_t i = 0; i < tiles_to_put_in_current_packet; i++) {
            uint32_t tile_id = tile_id_start + row_offset + pages_read_in_row;
            pages_read_in_row++;
            if (pages_read_in_row >= k_block_tiles) {
                row_offset += output_tensor_Wt;
                pages_read_in_row = 0;
            }
            noc_addrs[i] = tt::tt_fabric::linear::addrgen_detail::get_noc_address(output_addrgen, tile_id, 0);
        }
        if (tiles_to_put_in_current_packet > 1) {
            fabric_unicast_noc_scatter_write_with_state<
                UnicastScatterWriteUpdateMask::DstAddrs | UnicastScatterWriteUpdateMask::ChunkSizes |
                UnicastScatterWriteUpdateMask::PayloadSize>(
                mux_connection_handle,
                pkt_scatter_hdr,
                l1_read_addr,
                NocUnicastScatterCommandHeader(noc_addrs, chunk_sizes, tiles_to_put_in_current_packet),
                page_size * tiles_to_put_in_current_packet);
        } else {
            fabric_unicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                mux_connection_handle, pkt_unicast_hdr, l1_read_addr, NocUnicastCommandHeader{noc_addrs[0]});
        }
        tiles_read += tiles_to_put_in_current_packet;
        l1_read_addr += tiles_to_put_in_current_packet;
    }

    // unicast output ready semaphore
    fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
        mux_connection_handle,
        pkt_hdr_sem_inc,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt, 0});
}
#endif

bool is_backward_k_block_iter(uint32_t k_block_iter, uint32_t k_blocks_per_device) {
    // Start with self, then go backward, then forward
    uint32_t device_iter = k_block_iter / k_blocks_per_device;
    return (device_iter % 2);
}

bool is_injector(uint32_t use_backward, bool is_injector_core_backward, bool is_injector_core_forward) {
    return (is_injector_core_backward && use_backward) || (is_injector_core_forward && !use_backward);
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
