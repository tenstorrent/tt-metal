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

#ifdef IS_IN0
void compute_actual_k_block(
    uint32_t k_block_iter,
    uint32_t total_k_block_count,
    uint32_t my_rank,
    uint32_t k_blocks_per_device,
    uint32_t k_tiles_per_block,
    uint32_t num_devices,
    bool is_forward,
    bool is_first_n_block,
    volatile tt_l1_ptr uint32_t* out_ready_semaphore_forward,
    volatile tt_l1_ptr uint32_t* out_ready_semaphore_backward,
    uint32_t& sem_target_forward,
    uint32_t& sem_target_backward,
    bool is_injector_core,
    uint32_t in0_core_order_size,
    uint32_t& k_left_start_tile,
    uint32_t& k_right_start_tile) {
#else
void compute_actual_k_block(
    uint32_t k_block_iter,
    uint32_t total_k_block_count,
    uint32_t my_rank,
    uint32_t k_blocks_per_device,
    uint32_t k_tiles_per_block,
    uint32_t num_devices,
    bool is_forward,
    uint32_t& k_left_start_tile,
    uint32_t& k_right_start_tile) {
#endif
    // Start with self
    // Then for each device_iter, you are reading k_blocks_per_device half blocks from each direction
    // Left block coming from your forward device, and right block coming from your backward device
    uint32_t actual_k_block_iter = is_forward ? k_block_iter : (total_k_block_count - 1 - k_block_iter);
    uint32_t device_iter = actual_k_block_iter / k_blocks_per_device;
    uint32_t device_k_block_iter = actual_k_block_iter % k_blocks_per_device;
    if (device_iter == 0) {
        // Local
        k_left_start_tile = (my_rank * k_blocks_per_device + device_k_block_iter) * k_tiles_per_block;
        k_right_start_tile = k_left_start_tile + k_tiles_per_block / 2;
    } else {
        // Remote
        // Forward rank (origin of left half)
        int32_t actual_device_rank = my_rank + device_iter;
        if ((uint32_t)actual_device_rank >= num_devices) {
            actual_device_rank = actual_device_rank - num_devices;
        }
        k_left_start_tile = (actual_device_rank * k_blocks_per_device + device_k_block_iter) * k_tiles_per_block;

        // Backward rank
        actual_device_rank = my_rank - device_iter;
        if (actual_device_rank < 0) {
            actual_device_rank = num_devices + actual_device_rank;
        }
        k_right_start_tile = (actual_device_rank * k_blocks_per_device + device_k_block_iter) * k_tiles_per_block +
                             k_tiles_per_block / 2;
    }
#ifdef IS_IN0
    if (device_iter > 0 && is_first_n_block) {
        // When we are not reading from local, and we are in the first forward pass through n, wait for data to arrive
        if (is_injector_core) {
            noc_semaphore_wait_min(out_ready_semaphore_forward, sem_target_forward + in0_core_order_size);
            sem_target_forward += in0_core_order_size;
            noc_semaphore_wait_min(out_ready_semaphore_backward, sem_target_backward + in0_core_order_size);
            sem_target_backward += in0_core_order_size;
        }
    }
#endif
}

#ifdef USE_MUX
struct PacketHeaders {
    volatile tt::tt_fabric::LowLatencyPacketHeader* scatter_hdr;
    volatile tt::tt_fabric::LowLatencyPacketHeader* unicast_hdr;
    volatile tt::tt_fabric::LowLatencyPacketHeader* sem_inc_hdr;
};

template <typename TensorAccessorType, typename ConnectionHandleType>
void forward_half_block_to_fabric_neighbor(
    uint32_t m_tile_start,
    uint32_t k_tile_start,  // this is the k_tile_index in the output
    uint32_t m_block_tiles,
    uint32_t k_block_tiles,
    uint32_t num_tiles_to_write_per_packet,
    uint32_t in0_start_address,
    uint32_t output_tensor_Wt,
    const TensorAccessorType& output_addrgen,
    ConnectionHandleType mux_connection_handle,
    PacketHeaders pkt_hdrs,
    uint16_t page_size,
    uint64_t out_ready_sem_noc_addr_in_pkt,
    bool write_left_half,
    uint32_t m_tile_end) {
    auto* pkt_scatter_hdr = pkt_hdrs.scatter_hdr;
    auto* pkt_unicast_hdr = pkt_hdrs.unicast_hdr;
    auto* pkt_hdr_sem_inc = pkt_hdrs.sem_inc_hdr;

    uint32_t half_k_block_tiles = k_block_tiles / 2;
    uint32_t tiles_to_read = m_block_tiles * half_k_block_tiles;
    uint32_t tiles_read = 0;
    uint32_t tile_id_start = m_tile_start * output_tensor_Wt + k_tile_start;
    uint32_t row_offset = 0;
    uint32_t pages_read_in_row = 0;
    size_t l1_read_addr = in0_start_address + (write_left_half ? 0 : (half_k_block_tiles * page_size));
    while (tiles_read < tiles_to_read) {
        if (m_tile_start >= m_tile_end) {
            break;
        }
        uint32_t tiles_remaining_to_read = tiles_to_read - tiles_read;
        uint32_t tiles_to_put_in_current_packet = std::min(tiles_remaining_to_read, num_tiles_to_write_per_packet);
        bool reached_half_block_end = false;

        uint16_t chunk_sizes[3] = {page_size, page_size, page_size};
        uint64_t noc_addrs[4] = {0, 0, 0, 0};
        for (uint32_t i = 0; i < tiles_to_put_in_current_packet; i++) {
            uint32_t tile_id = tile_id_start + row_offset + pages_read_in_row;
            pages_read_in_row++;
            if (pages_read_in_row >= half_k_block_tiles) {
                row_offset += output_tensor_Wt;
                pages_read_in_row = 0;
                reached_half_block_end = true;
                tiles_to_put_in_current_packet = i + 1;  // break early because not contiguous in L1
                m_tile_start++;
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

        noc_async_writes_flushed();
        tiles_read += tiles_to_put_in_current_packet;
        l1_read_addr += (tiles_to_put_in_current_packet * page_size);
        if (reached_half_block_end) {
            l1_read_addr += (half_k_block_tiles * page_size);
        }
    }

    // unicast output ready semaphore
    fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
        mux_connection_handle,
        pkt_hdr_sem_inc,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt, 0});

    noc_async_writes_flushed();
}
#endif

bool is_backward_k_block_iter(uint32_t k_block_iter, uint32_t k_blocks_per_device) {
    // Start with self, then go backward, then forward
    uint32_t device_iter = k_block_iter / k_blocks_per_device;
    return (device_iter % 2);
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
    uint32_t d1_start_left,
    uint32_t d1_end_left,
    uint32_t d1_start_right,
    uint32_t d1_end_right) {
    ASSERT(d0_end > d0_start);
    ASSERT(d1_end_left > d1_start_left);
    ASSERT(d1_end_right > d1_start_right);

    for (uint32_t i = d0_start; i < d0_end; i++) {
        if (i >= shape.logical_d0) {
            break;
        }
        for (uint32_t j = d1_start_left; j < d1_end_left; j++) {
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
        write_ptr += (K_block_tiles / 2 - (d1_end_left - d1_start_left)) * tile_size_bytes;
        for (uint32_t j = d1_start_right; j < d1_end_right; j++) {
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
        write_ptr += (K_block_tiles / 2 - (d1_end_right - d1_start_right)) * tile_size_bytes;
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
    uint32_t d0_start_left,
    uint32_t d0_end_left,
    uint32_t d0_start_right,
    uint32_t d0_end_right,
    uint32_t d1_start,
    uint32_t d1_end) {
    ASSERT(d0_end_left > d0_start_left);
    ASSERT(d0_end_right > d0_start_right);
    ASSERT(d1_end > d1_start);
    for (uint32_t i = d0_start_left; i < d0_end_left; i++) {
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
        // finish up incrementing write_ptr if (d1_end - d1_start) < N_block_tiles
        write_ptr += (N_block_tiles - (d1_end - d1_start)) * tile_size_bytes;
    }
    for (uint32_t i = d0_start_right; i < d0_end_right; i++) {
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
        // finish up incrementing write_ptr if (d1_end - d1_start) < N_block_tiles
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

#ifdef USE_MUX
template <uint32_t NumBuffersPerChannel, uint32_t ChannelBufferSizeBytes>
struct MuxConnection {
    // Params
    bool connection_valid;
    bool is_termination_master;
    uint8_t fabric_mux_x;
    uint8_t fabric_mux_y;
    size_t fabric_mux_channel_base_address;
    size_t fabric_mux_connection_info_address;
    size_t fabric_mux_connection_handshake_address;
    size_t fabric_mux_flow_control_address;
    size_t fabric_mux_buffer_index_address;
    uint8_t fabric_mux_channel_id;

    uint32_t termination_sync_address;
    uint32_t local_fabric_mux_status_address;
    uint32_t local_flow_control_address;
    uint32_t local_teardown_address;
    uint32_t local_buffer_index_address;

    uint32_t termination_master_noc_x;
    uint32_t termination_master_noc_y;

    // Connection state
    tt::tt_fabric::WorkerToFabricMuxSender<NumBuffersPerChannel> connection;

    FORCE_INLINE tt::tt_fabric::WorkerToFabricMuxSender<NumBuffersPerChannel>* build_and_connect(
        size_t fabric_mux_status_address) {
        if (!connection_valid) {
            return nullptr;
        }

        connection = tt::tt_fabric::build_connection_to_fabric_endpoint<NumBuffersPerChannel>(
            fabric_mux_x,
            fabric_mux_y,
            fabric_mux_channel_id,
            NumBuffersPerChannel,
            ChannelBufferSizeBytes,
            fabric_mux_channel_base_address,
            fabric_mux_connection_info_address,
            fabric_mux_connection_handshake_address,
            fabric_mux_flow_control_address,
            fabric_mux_buffer_index_address,
            local_flow_control_address,
            local_teardown_address,
            local_buffer_index_address);

        tt::tt_fabric::wait_for_fabric_endpoint_ready(
            fabric_mux_x, fabric_mux_y, fabric_mux_status_address, local_fabric_mux_status_address);

        tt::tt_fabric::fabric_client_connect(connection);

        return &connection;
    }
};

template <uint32_t NumBuffersPerChannel, uint32_t ChannelBufferSizeBytes>
FORCE_INLINE MuxConnection<NumBuffersPerChannel, ChannelBufferSizeBytes> parse_mux_connection_args(
    uint32_t& argidx, uint32_t in0_core_order_index, uint32_t valid_core_index) {
    MuxConnection<NumBuffersPerChannel, ChannelBufferSizeBytes> mux;

    mux.connection_valid = ((get_arg_val<uint32_t>(argidx++) == 1) && (in0_core_order_index == valid_core_index));
    mux.is_termination_master = get_arg_val<uint32_t>(argidx++);
    mux.fabric_mux_x = get_arg_val<uint32_t>(argidx++);
    mux.fabric_mux_y = get_arg_val<uint32_t>(argidx++);
    mux.fabric_mux_channel_base_address = get_arg_val<uint32_t>(argidx++);
    mux.fabric_mux_connection_info_address = get_arg_val<uint32_t>(argidx++);
    mux.fabric_mux_connection_handshake_address = get_arg_val<uint32_t>(argidx++);
    mux.fabric_mux_flow_control_address = get_arg_val<uint32_t>(argidx++);
    mux.fabric_mux_buffer_index_address = get_arg_val<uint32_t>(argidx++);
    mux.fabric_mux_channel_id = get_arg_val<uint32_t>(argidx++);

    mux.termination_sync_address = get_semaphore(get_arg_val<uint32_t>(argidx++));
    mux.local_fabric_mux_status_address = get_semaphore(get_arg_val<uint32_t>(argidx++));
    mux.local_flow_control_address = get_semaphore(get_arg_val<uint32_t>(argidx++));
    mux.local_teardown_address = get_semaphore(get_arg_val<uint32_t>(argidx++));
    mux.local_buffer_index_address = get_semaphore(get_arg_val<uint32_t>(argidx++));

    mux.termination_master_noc_x = get_arg_val<uint32_t>(argidx++);
    mux.termination_master_noc_y = get_arg_val<uint32_t>(argidx++);

    return mux;
}

template <typename AddrGen, typename RouteInfo>
FORCE_INLINE PacketHeaders allocate_and_init_packet_headers(
    bool valid,
    const RouteInfo& unicast_route_info,
    const AddrGen& in0_reader,
    uint32_t num_tiles_to_write_per_packet,
    uint32_t in3_tile_size) {
    PacketHeaders hdrs;
    hdrs.scatter_hdr = PacketHeaderPool::allocate_header();
    hdrs.unicast_hdr = PacketHeaderPool::allocate_header();
    hdrs.sem_inc_hdr = PacketHeaderPool::allocate_header();

    if (valid) {
        uint16_t page_size = tt::tt_fabric::linear::addrgen_detail::get_page_size(in0_reader);
        uint64_t dummy_addrs[4] = {0, 0, 0, 0};
        uint16_t chunk_sizes[3] = {page_size, page_size, page_size};

        fabric_unicast_noc_scatter_write_set_state<
            UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
            hdrs.scatter_hdr,
            static_cast<uint8_t>(unicast_route_info.distance_in_hops),
            NocUnicastScatterCommandHeader(dummy_addrs, chunk_sizes, num_tiles_to_write_per_packet),
            page_size * num_tiles_to_write_per_packet);

        fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
            hdrs.unicast_hdr, static_cast<uint8_t>(unicast_route_info.distance_in_hops), nullptr, in3_tile_size);

        fabric_unicast_noc_unicast_atomic_inc_set_state<
            UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
            hdrs.sem_inc_hdr,
            static_cast<uint8_t>(unicast_route_info.distance_in_hops),
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0, static_cast<uint32_t>(1)});

        ccl_routing_utils::fabric_set_line_unicast_route(hdrs.scatter_hdr, unicast_route_info);
        ccl_routing_utils::fabric_set_line_unicast_route(hdrs.unicast_hdr, unicast_route_info);
        ccl_routing_utils::fabric_set_line_unicast_route(hdrs.sem_inc_hdr, unicast_route_info);
    }

    return hdrs;
}

template <typename ConnectionHandleType>
void close_mux(
    ConnectionHandleType mux_connection_handle,
    bool is_termination_master,
    uint32_t termination_sync_address,
    uint32_t num_mux_clients,
    const uint8_t fabric_mux_x,
    const uint8_t fabric_mux_y,
    size_t fabric_mux_termination_signal_address,
    uint32_t termination_master_noc_x,
    uint32_t termination_master_noc_y) {
    tt::tt_fabric::fabric_client_disconnect(*mux_connection_handle);
    if (is_termination_master) {
        auto* termination_sync_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_sync_address);
        noc_semaphore_wait(termination_sync_ptr, num_mux_clients - 1);
        tt::tt_fabric::fabric_endpoint_terminate(fabric_mux_x, fabric_mux_y, fabric_mux_termination_signal_address);
    } else {
        uint64_t dest_addr =
            safe_get_noc_addr(termination_master_noc_x, termination_master_noc_y, termination_sync_address, 0);
        noc_semaphore_inc(dest_addr, 1);
        noc_async_atomic_barrier();
    }
}
#endif
