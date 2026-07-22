// SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <algorithm>
#include <tuple>
#include <utility>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"

namespace detail {
template <typename... Args, uint32_t... Indexes>
auto make_tensor_accessor_tuple_impl(
    const std::tuple<Args...>& args_tuple,
    uint32_t address_rt_arg_index_start,
    uint32_t page_size,
    std::integer_sequence<uint32_t, Indexes...>) {
    // Third argument page_size from runtime args overrides TensorAccessorArgs::AlignedPageSize, which may be stale on
    // program cache hits.
    return std::make_tuple(TensorAccessor(
        std::get<Indexes>(args_tuple), get_arg_val<uint32_t>(address_rt_arg_index_start + Indexes), page_size)...);
}

template <typename... Args, uint32_t... Indexes>
auto make_tensor_accessor_tuple_impl_common(
    const std::tuple<Args...>& args_tuple,
    uint32_t common_arg_index_start,
    uint32_t page_size,
    std::integer_sequence<uint32_t, Indexes...>) {
    // Third argument page_size from runtime args overrides TensorAccessorArgs::AlignedPageSize, which may be stale on
    // program cache hits.
    return std::make_tuple(TensorAccessor(
        std::get<Indexes>(args_tuple), get_common_arg_val<uint32_t>(common_arg_index_start + Indexes), page_size)...);
}
}  // namespace detail

/**
 * Create a tuple of TensorAccessors from a tuple of TensorAccessorArgs.
 * Each tensor gets its address from consecutive RT args starting at address_rt_arg_index_start.
 */
template <typename... Args>
auto make_tensor_accessor_tuple_uniform_page_size(
    const std::tuple<Args...>& args_tuple, uint32_t address_rt_arg_index_start, uint32_t page_size) {
    return detail::make_tensor_accessor_tuple_impl(
        args_tuple, address_rt_arg_index_start, page_size, std::make_integer_sequence<uint32_t, sizeof...(Args)>());
}

/**
 * Same as make_tensor_accessor_tuple_uniform_page_size but reads addresses from common runtime args.
 */
template <typename... Args>
auto make_tensor_accessor_tuple_uniform_page_size_common(
    const std::tuple<Args...>& args_tuple, uint32_t common_arg_index_start, uint32_t page_size) {
    return detail::make_tensor_accessor_tuple_impl_common(
        args_tuple, common_arg_index_start, page_size, std::make_integer_sequence<uint32_t, sizeof...(Args)>());
}

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

#if defined(IS_IN0) || defined(IS_IN1)
template <bool IsLinear>
void compute_actual_k_block(
    uint32_t k_block_iter,
    uint32_t total_k_block_count,
    uint32_t my_rank,
    uint32_t k_blocks_per_device,
    uint32_t k_tiles_per_block,
    uint32_t k_tiles_per_device,
    uint32_t num_devices,
    bool is_forward,
    bool wait_for_forwarded_data,
    volatile tt_l1_ptr uint32_t* out_ready_semaphore_forward,
    volatile tt_l1_ptr uint32_t* out_ready_semaphore_backward,
    uint32_t& sem_target_forward,
    uint32_t& sem_target_backward,
    bool is_injector_core,
    uint32_t core_order_size,
    uint32_t k_left_tiles,
    uint32_t& k_left_start_tile,
    uint32_t& k_right_start_tile) {
#else
template <bool IsLinear>
void compute_actual_k_block(
    uint32_t k_block_iter,
    uint32_t total_k_block_count,
    uint32_t my_rank,
    uint32_t k_blocks_per_device,
    uint32_t k_tiles_per_block,
    uint32_t k_tiles_per_device,
    uint32_t num_devices,
    bool is_forward,
    uint32_t k_left_tiles,
    uint32_t& k_left_start_tile,
    uint32_t& k_right_start_tile) {
#endif
    // Start with self
    // Then for each device_iter, read k_blocks_per_device blocks from each direction.
    // Ring: left half from forward device, right half from backward device (bidirectional half-block).
    // Linear: full block from one direction (k_left=K_block_tiles, k_right=0), relayed hop-by-hop.
    //
    // K-tile address = device_rank * k_tiles_per_device + device_k_block_iter * k_tiles_per_block.
    // k_tiles_per_device is the *actual* K-tile span per device (not k_blocks_per_device *
    // k_tiles_per_block, which over-counts by k_tiles_per_block - K_block_tail_tiles when the
    // last block per device is a tail block).
    uint32_t actual_k_block_iter = is_forward ? k_block_iter : (total_k_block_count - 1 - k_block_iter);
    uint32_t device_iter = actual_k_block_iter / k_blocks_per_device;
    uint32_t device_k_block_iter = actual_k_block_iter % k_blocks_per_device;
    if (device_iter == 0) {
        // Local
        k_left_start_tile = my_rank * k_tiles_per_device + device_k_block_iter * k_tiles_per_block;
        k_right_start_tile = k_left_start_tile + k_left_tiles;
    } else {
        if constexpr (IsLinear) {
            // Linear uni-ring: slice at iter K = (my_rank + K) mod N. Data flows leftward
            // around a virtual ring (Dev k -> Dev k-1 each iter; Dev 0 long-sends to Dev N-1).
            uint32_t actual_device_rank = my_rank + device_iter;
            if (actual_device_rank >= num_devices) {
                actual_device_rank -= num_devices;
            }
            k_left_start_tile = actual_device_rank * k_tiles_per_device + device_k_block_iter * k_tiles_per_block;
            k_right_start_tile = k_left_start_tile;  // unused: k_right_tiles == 0
        } else {
            // Ring: bidirectional half-block. Left half from forward device, right half from
            // backward device (modular wrap-around).
            int32_t actual_device_rank = my_rank + device_iter;
            if ((uint32_t)actual_device_rank >= num_devices) {
                actual_device_rank = actual_device_rank - num_devices;
            }
            k_left_start_tile = actual_device_rank * k_tiles_per_device + device_k_block_iter * k_tiles_per_block;

            actual_device_rank = my_rank - device_iter;
            if (actual_device_rank < 0) {
                actual_device_rank = num_devices + actual_device_rank;
            }
            k_right_start_tile =
                actual_device_rank * k_tiles_per_device + device_k_block_iter * k_tiles_per_block + k_left_tiles;
        }
    }
#if defined(IS_IN0) || defined(IS_IN1)
    if (wait_for_forwarded_data) {
        if (is_injector_core) {
            if constexpr (IsLinear) {
                // Linear uni-ring: one slice per iter from "successor" (Dev k+1 normally; for
                // Dev N-1, from Dev 0 via long send). All sends use out_ready_semaphore_forward
                // at the receiver -- single sem per iter.
                //
                // The sender skips the redundant final relay lap (the last K_blocks_per_device
                // iters; see dm_in0_sender), so it fires exactly (N-1)*K_blocks_per_device sem
                // incs per m_block -- precisely the count the receiver waits on here. sem ends each
                // m_block exactly at sem_target with nothing left in flight, so no compensation
                // (and no end-of-op drain) is needed.
                if (device_iter > 0) {
                    noc_semaphore_wait_min(out_ready_semaphore_forward, sem_target_forward + 1);
                    sem_target_forward += 1;
                }
            } else if (device_iter > 0) {
                // Ring: both halves arrive simultaneously from both directions
                // (both neighbors always exist for Ring topology by construction).
                noc_semaphore_wait_min(out_ready_semaphore_forward, sem_target_forward + core_order_size);
                sem_target_forward += core_order_size;
                noc_semaphore_wait_min(out_ready_semaphore_backward, sem_target_backward + core_order_size);
                sem_target_backward += core_order_size;
            }
        }
    }
#endif  // IS_IN0 || IS_IN1
}

#ifdef USE_MUX
struct PacketHeaders {
    volatile tt::tt_fabric::LowLatencyPacketHeader* scatter_hdr;
    volatile tt::tt_fabric::LowLatencyPacketHeader* unicast_hdr;
    volatile tt::tt_fabric::LowLatencyPacketHeader* sem_inc_hdr;
};

template <typename TensorAccessorType, typename ConnectionHandleType>
FORCE_INLINE void forward_half_block_to_fabric_neighbor(
    Noc noc,
    uint32_t m_tile_start,
    uint32_t k_tile_start,  // this is the k_tile_index in the output
    uint32_t m_block_tiles,
    uint32_t k_left_tiles,
    uint32_t k_right_tiles,
    uint32_t num_tiles_to_write_per_packet,
    uint32_t in0_start_address,
    uint32_t output_tensor_Wt,
    const TensorAccessorType& output_addrgen,
    ConnectionHandleType mux_connection_handle,
    PacketHeaders pkt_hdrs,
    uint16_t page_size,
    uint64_t out_ready_sem_noc_addr_in_pkt,
    bool write_left_half,
    uint32_t m_tile_end,
    bool do_write) {
    auto* pkt_scatter_hdr = pkt_hdrs.scatter_hdr;
    auto* pkt_unicast_hdr = pkt_hdrs.unicast_hdr;
    auto* pkt_hdr_sem_inc = pkt_hdrs.sem_inc_hdr;

    uint32_t half_k_block_tiles = write_left_half ? k_left_tiles : k_right_tiles;
    uint32_t other_half_k_block_tiles = write_left_half ? k_right_tiles : k_left_tiles;
    uint32_t tiles_to_read = m_block_tiles * half_k_block_tiles;
    uint32_t tiles_read = 0;
    uint32_t tile_id_start = m_tile_start * output_tensor_Wt + k_tile_start;
    uint32_t row_offset = 0;
    uint32_t pages_read_in_row = 0;
    size_t l1_read_addr = in0_start_address + (write_left_half ? 0 : (k_left_tiles * page_size));
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
        if (do_write) {
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

            noc.async_writes_flushed();
            tiles_read += tiles_to_put_in_current_packet;
            l1_read_addr += (tiles_to_put_in_current_packet * page_size);
            if (reached_half_block_end) {
                l1_read_addr += (other_half_k_block_tiles * page_size);
            }
        }
    }

    // unicast output ready semaphore
    fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
        mux_connection_handle,
        pkt_hdr_sem_inc,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt, 0});

    noc.async_writes_flushed();
}

// Sibling of forward_half_block_to_fabric_neighbor for in1 (FSDP weight gather).
// Splits the same K dimension as in0 so left/right halves stay K-aligned across both operands,
// but walks (half_k_block_tiles rows x N_block_tiles per-row tiles) instead of (m_block_tiles x half_k).
// In1 L1 layout is K_block_tiles rows x N_block_tiles cols of tiles; PWB layout is [K_full, N_local].
template <typename TensorAccessorType, typename ConnectionHandleType>
FORCE_INLINE void forward_in1_half_block_to_fabric_neighbor(
    Noc noc,
    uint32_t k_tile_start,  // K-tile index in PWB at the start of the full K-block
    uint32_t n_tile_start,  // N-tile index in PWB at the start of the current N-block
    uint32_t k_left_tiles,
    uint32_t k_right_tiles,
    uint32_t current_N_block_tiles,  // actual N tiles to write per row (handles partial last-N)
    uint32_t N_block_tiles,          // padded N tiles per K-row in the L1 block (for L1 stride)
    uint32_t num_tiles_to_write_per_packet,
    uint32_t in1_start_address,
    uint32_t pwb_N_Wt,  // PWB N width in tiles
    const TensorAccessorType& pwb_addrgen,
    ConnectionHandleType mux_connection_handle,
    PacketHeaders pkt_hdrs,
    uint16_t page_size,
    uint64_t out_ready_sem_noc_addr_in_pkt,
    bool write_left_half,
    bool do_write) {
    auto* pkt_scatter_hdr = pkt_hdrs.scatter_hdr;
    auto* pkt_unicast_hdr = pkt_hdrs.unicast_hdr;
    auto* pkt_hdr_sem_inc = pkt_hdrs.sem_inc_hdr;

    uint32_t half_k_block_tiles = write_left_half ? k_left_tiles : k_right_tiles;
    uint32_t k_half_offset = write_left_half ? 0 : k_left_tiles;  // K-tile offset within the K-block
    uint32_t tiles_to_read = half_k_block_tiles * current_N_block_tiles;
    uint32_t tiles_read = 0;
    uint32_t k_row_in_half = 0;
    uint32_t col_in_row = 0;
    // L1: each K-row stride is N_block_tiles * page_size. Skip the left K-half rows if writing the right half.
    size_t l1_read_addr = in1_start_address + (write_left_half ? 0 : (k_left_tiles * N_block_tiles * page_size));

    while (tiles_read < tiles_to_read) {
        uint32_t tiles_remaining_to_read = tiles_to_read - tiles_read;
        uint32_t tiles_to_put_in_current_packet = std::min(tiles_remaining_to_read, num_tiles_to_write_per_packet);
        bool reached_row_end = false;

        uint16_t chunk_sizes[3] = {page_size, page_size, page_size};
        uint64_t noc_addrs[4] = {0, 0, 0, 0};
        for (uint32_t i = 0; i < tiles_to_put_in_current_packet; i++) {
            uint32_t tile_id = (k_tile_start + k_half_offset + k_row_in_half) * pwb_N_Wt + (n_tile_start + col_in_row);
            col_in_row++;
            if (col_in_row >= current_N_block_tiles) {
                col_in_row = 0;
                k_row_in_half++;
                reached_row_end = true;
                tiles_to_put_in_current_packet = i + 1;  // break early at row boundary
            }
            noc_addrs[i] = tt::tt_fabric::linear::addrgen_detail::get_noc_address(pwb_addrgen, tile_id, 0);
        }
        if (do_write) {
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

            noc.async_writes_flushed();
            tiles_read += tiles_to_put_in_current_packet;
            l1_read_addr += (tiles_to_put_in_current_packet * page_size);
            if (reached_row_end) {
                // Skip the padded N tiles in L1 (between current_N_block_tiles and N_block_tiles).
                l1_read_addr += ((N_block_tiles - current_N_block_tiles) * page_size);
            }
        }
    }

    fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
        mux_connection_handle,
        pkt_hdr_sem_inc,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt, 0});

    noc.async_writes_flushed();
}
#endif

bool is_backward_k_block_iter(uint32_t k_block_iter, uint32_t k_blocks_per_device) {
    // Start with self, then go backward, then forward
    uint32_t device_iter = k_block_iter / k_blocks_per_device;
    return (device_iter % 2);
}

inline void fill_zeros_async(Noc noc, CircularBuffer cb, uint32_t bytes, uint32_t offset_bytes = 0) {
    noc.async_write_zeros(cb, bytes, {.offset_bytes = offset_bytes});
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
    Noc noc,
    const TensorAccessorType& tensor_accessor,
    const TensorShape2D& shape,
    CircularBuffer cb,
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
    uint32_t d1_tiles_left,
    uint32_t d1_start_right,
    uint32_t d1_end_right,
    uint32_t d1_tiles_right) {
    ASSERT(d0_end > d0_start);
    ASSERT(d1_end_left > d1_start_left);
    // Linear topology is unidirectional: the "right" (backward) half is legitimately empty
    // (k_right_tiles == 0 for a full block), so allow an empty range here. A non-empty-but-
    // inverted range would still be a bug, hence >=.
    ASSERT(d1_end_right >= d1_start_right);

    const uint32_t cb_base_write_ptr = cb.get_write_ptr();
    uint32_t write_ptr = cb_base_write_ptr;
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
                    noc.async_read(
                        in3_accessor, CoreLocalMem<uint8_t>(write_ptr), tile_size_bytes, {.page_id = tile_id}, {});
                } else {
#endif
                    uint32_t tile_id = i * shape.logical_d1 + j;
                    noc.async_read(
                        tensor_accessor, CoreLocalMem<uint8_t>(write_ptr), tile_size_bytes, {.page_id = tile_id}, {});
#ifdef READ_FROM_LOCAL_INPUT
                }
#endif
            } else {
                fill_zeros_async(noc, cb, tile_size_bytes, write_ptr - cb_base_write_ptr);
            }
            write_ptr += tile_size_bytes;
        }
        // finish up incrementing write_ptr if (d1_end - d1_start) < K_block_tiles
        write_ptr += (d1_tiles_left - (d1_end_left - d1_start_left)) * tile_size_bytes;
        for (uint32_t j = d1_start_right; j < d1_end_right; j++) {
            if (j < shape.logical_d1) {
#ifdef READ_FROM_LOCAL_INPUT
                if (local_k_start <= j && j <= local_k_end) {
                    // read from self_tensor_accessor
                    uint32_t tile_id = i * input_tensor_Wt + (j - local_k_start);
                    noc.async_read(
                        in3_accessor, CoreLocalMem<uint8_t>(write_ptr), tile_size_bytes, {.page_id = tile_id}, {});
                } else {
#endif
                    uint32_t tile_id = i * shape.logical_d1 + j;
                    noc.async_read(
                        tensor_accessor, CoreLocalMem<uint8_t>(write_ptr), tile_size_bytes, {.page_id = tile_id}, {});
#ifdef READ_FROM_LOCAL_INPUT
                }
#endif
            } else {
                fill_zeros_async(noc, cb, tile_size_bytes, write_ptr - cb_base_write_ptr);
            }
            write_ptr += tile_size_bytes;
        }
        // finish up incrementing write_ptr if (d1_end - d1_start) < K_block_tiles
        write_ptr += (d1_tiles_right - (d1_end_right - d1_start_right)) * tile_size_bytes;
    }
    noc.async_read_barrier();
    noc.write_zeros_l1_barrier();
}

/**
 * Read a block of in1 from a potentially padded tensor.
 * Since this is for matmul, no need to read when N >= logical_N
 * Otherwise, if K >= logical_K, fill with zeros.
 */
template <uint32_t K_block_tiles, uint32_t N_block_tiles, typename TensorAccessorType>
void read_in1_block_sync(
    Noc noc,
    const TensorAccessorType& tensor_accessor,
    const TensorShape2D& shape,
    CircularBuffer cb,
    uint32_t tile_size_bytes,
    uint32_t d0_start_left,
    uint32_t d0_end_left,
    uint32_t d0_start_right,
    uint32_t d0_end_right,
    uint32_t d1_start,
    uint32_t d1_end) {
    ASSERT(d0_end_left > d0_start_left);
    // Linear topology is unidirectional: the "right" (backward) half is legitimately empty.
    ASSERT(d0_end_right >= d0_start_right);
    ASSERT(d1_end > d1_start);
    const uint32_t cb_base_write_ptr = cb.get_write_ptr();
    uint32_t write_ptr = cb_base_write_ptr;
    for (uint32_t i = d0_start_left; i < d0_end_left; i++) {
        for (uint32_t j = d1_start; j < d1_end; j++) {
            if (j >= shape.logical_d1) {
                write_ptr += tile_size_bytes;
                continue;
            }
            if (i < shape.logical_d0) {
                uint32_t tile_id = i * shape.logical_d1 + j;
                noc.async_read(
                    tensor_accessor, CoreLocalMem<uint8_t>(write_ptr), tile_size_bytes, {.page_id = tile_id}, {});
            } else {
                fill_zeros_async(noc, cb, tile_size_bytes, write_ptr - cb_base_write_ptr);
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
                noc.async_read(
                    tensor_accessor, CoreLocalMem<uint8_t>(write_ptr), tile_size_bytes, {.page_id = tile_id}, {});
            } else {
                fill_zeros_async(noc, cb, tile_size_bytes, write_ptr - cb_base_write_ptr);
            }
            write_ptr += tile_size_bytes;
        }
        // finish up incrementing write_ptr if (d1_end - d1_start) < N_block_tiles
        write_ptr += (N_block_tiles - (d1_end - d1_start)) * tile_size_bytes;
    }
    noc.async_read_barrier();
    noc.write_zeros_l1_barrier();
}

/**
 * Write a block of output to a potentially padded tensor.
 * Skip writing when M >= logical_M or N >= logical_N
 */
template <uint32_t M_block_tiles, uint32_t N_block_tiles, typename TensorAccessorType>
void write_block_sync(
    Noc noc,
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
            noc.async_write(
                CoreLocalMem<uint8_t>(read_ptr), tensor_accessor, tile_size_bytes, {}, {.page_id = tile_id});
            read_ptr += tile_size_bytes;
        }
        // finish up incrementing read_ptr if (d1_end - d1_start) < N_block_tiles
        read_ptr += (N_block_tiles - (d1_end - d1_start)) * tile_size_bytes;
    }
    noc.async_writes_flushed();
}

/**
 * Read ternary inputs (ternary_a and ternary_b) and write data to CB
 *
 * For ternary_a: read M_block_tiles * N_block_tiles tiles (full block), pushed one row at a time.
 * For ternary_b:
 *   - broadcast_ternary_b=1: read 1 row of tiles (N_block_tiles), compute broadcasts across M rows.
 *   - broadcast_ternary_b=0: read M rows of tiles, pushed one row at a time (matches ternary_a pattern).
 *
 * Performance optimization: Unlike read_in0_block_sync and read_in1_block_sync, pushes ternary_a
 * tiles one row at a time. This allows the compute kernel to begin processing addcmul operations
 * as soon as the first row is ready, rather than waiting for the entire block. This overlapping
 * of data movement and compute improves overall throughput.
 */
template <uint32_t M_block_tiles, uint32_t N_block_tiles, typename TensorAccessorType>
void read_ternary_blocks_sync(
    Noc noc,
    const TensorAccessorType& ternary_a_accessor,
    const TensorAccessorType& ternary_b_accessor,
    const TensorShape2D& shape,
    CircularBuffer ternary_a_cb,
    CircularBuffer ternary_b_cb,
    uint32_t a_tile_size_bytes,
    uint32_t b_tile_size_bytes,
    uint32_t broadcast_ternary_b,
    uint32_t d0_start,
    uint32_t d0_end,
    uint32_t d1_start,
    uint32_t d1_end) {
    ASSERT(d0_end > d0_start);
    ASSERT(d1_end > d1_start);

    if (broadcast_ternary_b) {
        // Broadcast: read single row, push all at once
        ternary_b_cb.reserve_back(N_block_tiles);
        uint32_t ternary_b_write_ptr = ternary_b_cb.get_write_ptr();
        for (uint32_t n_tile_id = d1_start; n_tile_id < d1_end; n_tile_id++) {
            if (n_tile_id >= shape.logical_d1) {
                break;
            }
            noc.async_read(
                ternary_b_accessor,
                CoreLocalMem<uint8_t>(ternary_b_write_ptr),
                b_tile_size_bytes,
                {.page_id = n_tile_id},
                {});
            ternary_b_write_ptr += b_tile_size_bytes;
        }
        noc.async_read_barrier();
        ternary_b_cb.push_back(N_block_tiles);
    } else {
        // No broadcast: read row-by-row (matches ternary_a pattern)
        uint32_t b_m_id = 0;
        uint32_t b_i = d0_start;
        for (; b_i < d0_end; b_i++, b_m_id++) {
            ternary_b_cb.reserve_back(N_block_tiles);
            uint32_t ternary_b_write_ptr = ternary_b_cb.get_write_ptr();
            for (uint32_t j = d1_start; j < d1_end; j++) {
                if (j >= shape.logical_d1) {
                    break;
                }
                if (b_i < shape.logical_d0) {
                    uint32_t tile_id = b_i * shape.logical_d1 + j;
                    noc.async_read(
                        ternary_b_accessor,
                        CoreLocalMem<uint8_t>(ternary_b_write_ptr),
                        b_tile_size_bytes,
                        {.page_id = tile_id},
                        {});
                }
                ternary_b_write_ptr += b_tile_size_bytes;
            }
            noc.async_read_barrier();
            ternary_b_cb.push_back(N_block_tiles);
        }
        for (; b_m_id < M_block_tiles; b_m_id++) {
            ternary_b_cb.reserve_back(N_block_tiles);
            ternary_b_cb.push_back(N_block_tiles);
        }
    }

    uint32_t m_id = 0;
    uint32_t i = d0_start;
    for (; i < d0_end; i++, m_id++) {
        ternary_a_cb.reserve_back(N_block_tiles);

        uint32_t ternary_a_write_ptr = ternary_a_cb.get_write_ptr();
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
                noc.async_read(
                    ternary_a_accessor,
                    CoreLocalMem<uint8_t>(ternary_a_write_ptr),
                    a_tile_size_bytes,
                    {.page_id = tile_id},
                    {});
            }
            ternary_a_write_ptr += a_tile_size_bytes;
        }
        noc.async_read_barrier();

        ternary_a_cb.push_back(N_block_tiles);
    }
    for (; m_id < M_block_tiles; m_id++) {
        ternary_a_cb.reserve_back(N_block_tiles);
        ternary_a_cb.push_back(N_block_tiles);
    }
}

/**
 * This write method is more granular, waiting on a row of output tiles
 * in the output CB before writing those out, rather than waiting on the entire block.
 */
template <uint32_t M_block_tiles, uint32_t N_block_tiles, typename TensorAccessorType>
void write_block_sync_granular(
    Noc noc,
    const TensorAccessorType& tensor_accessor,
    const TensorShape2D& shape,
    CircularBuffer cb_out,
    uint32_t tile_size_bytes,
    uint32_t d0_start,
    uint32_t d0_end,
    uint32_t d1_start,
    uint32_t d1_end) {
    for (uint32_t m_id = 0; m_id < M_block_tiles; m_id++) {
        cb_out.wait_front(N_block_tiles);
        uint32_t m_tile = d0_start + m_id;
        if (m_tile < d0_end && m_tile < shape.logical_d0) {
            uint32_t out_read_ptr = cb_out.get_read_ptr();
            for (uint32_t n_tile_id = d1_start; n_tile_id < d1_end; n_tile_id++) {
                if (n_tile_id >= shape.logical_d1) {
                    break;
                }
                uint32_t tile_id = m_tile * shape.logical_d1 + n_tile_id;
                noc.async_write(
                    CoreLocalMem<uint8_t>(out_read_ptr), tensor_accessor, tile_size_bytes, {}, {.page_id = tile_id});
                out_read_ptr += tile_size_bytes;
            }
        }
        cb_out.pop_front(N_block_tiles);
    }
    noc.async_writes_flushed();
}

/**
 * Helper: dispatch to correct tuple element using fold expression.
 * Each branch calls noc.async_write with the concrete TensorAccessor type.
 */
template <typename Tuple, size_t... Is>
FORCE_INLINE void write_tile_to_chunk(
    Noc noc,
    const Tuple& accessors,
    uint32_t chunk_idx,
    uint32_t tile_id,
    uint32_t read_ptr,
    uint32_t tile_size_bytes,
    std::index_sequence<Is...>) {
    // Fold expression: expands to if/else chain at compile time
    // Each branch calls noc.async_write with the concrete TensorAccessor type
    ((chunk_idx == Is
          ? (noc.async_write(
                 CoreLocalMem<uint8_t>(read_ptr), std::get<Is>(accessors), tile_size_bytes, {}, {.page_id = tile_id}),
             void())
          : void()),
     ...);
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
    Noc noc,
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
                noc,
                accessors,
                chunk_idx,
                tile_id_in_chunk,
                read_ptr,
                tile_size_bytes,
                std::index_sequence_for<Accessors...>{});
            read_ptr += tile_size_bytes;
        }
        // finish up incrementing read_ptr if (d1_end - d1_start) < N_block_tiles
        read_ptr += (N_block_tiles - (d1_end - d1_start)) * tile_size_bytes;
    }
    noc.async_writes_flushed();
}

/**
 * Variadic write method for split operation with N output tensors.
 * Takes the tuple directly, preserving concrete TensorAccessor<DSpec> types for noc.async_write.
 */
template <
    uint32_t M_block_tiles,
    uint32_t N_block_tiles,
    uint32_t N_chunks,
    uint32_t N_tiles_per_chunk,
    typename... Accessors>
void write_block_sync_granular_split(
    Noc noc,
    const std::tuple<Accessors...>& accessors,
    const TensorShape2D& chunk_shape,
    CircularBuffer cb_out,
    uint32_t tile_size_bytes,
    uint32_t d0_start,
    uint32_t d0_end,
    uint32_t d1_start,
    uint32_t d1_end) {
    const uint32_t chunk_idx_start = d1_start / N_tiles_per_chunk;
    const uint32_t tile_idx_in_chunk_start = d1_start % N_tiles_per_chunk;

    for (uint32_t m_id = 0; m_id < M_block_tiles; m_id++) {
        cb_out.wait_front(N_block_tiles);
        uint32_t m_tile = d0_start + m_id;
        if (m_tile < d0_end && m_tile < chunk_shape.logical_d0) {
            uint32_t out_read_ptr = cb_out.get_read_ptr();

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
                    noc,
                    accessors,
                    chunk_idx,
                    tile_id,
                    out_read_ptr,
                    tile_size_bytes,
                    std::index_sequence_for<Accessors...>{});

                out_read_ptr += tile_size_bytes;
            }
        }
        cb_out.pop_front(N_block_tiles);
    }
    noc.async_writes_flushed();
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

    // Actual number of worker clients on THIS mux (the termination master waits for
    // num_mux_clients-1 peers). Per-mux, not a blanket num_workers_per_link: when the sender
    // axis isn't a multiple of num_workers_per_link the last group is short, so this can be < nwpl.
    uint32_t num_mux_clients;

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
    mux.num_mux_clients = get_arg_val<uint32_t>(argidx++);

    return mux;
}

template <typename AddrGen, typename RouteInfo>
FORCE_INLINE PacketHeaders allocate_and_init_packet_headers(
    bool valid,
    const RouteInfo& unicast_route_info,
    const AddrGen& in0_reader,
    uint32_t num_tiles_to_write_per_packet,
    uint32_t in3_tile_size) {
    PacketHeaders hdrs{};  // zero-initialized (nullptrs) when !valid
    if (!valid) {
        return hdrs;
    }
    hdrs.scatter_hdr = PacketHeaderPool::allocate_header();
    hdrs.unicast_hdr = PacketHeaderPool::allocate_header();
    hdrs.sem_inc_hdr = PacketHeaderPool::allocate_header();

    {
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
FORCE_INLINE void close_mux(
    Noc noc,
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
        noc.async_atomic_barrier();
    }
}
#endif
