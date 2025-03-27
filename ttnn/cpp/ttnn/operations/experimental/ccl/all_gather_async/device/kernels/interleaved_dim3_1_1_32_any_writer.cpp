// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_constants.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/noc_addr.hpp"
#include "minimal_ccl_common.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using tt::tt_metal::BufferType;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t reserved_packet_header_cb_id = get_compile_time_arg_val(1);
constexpr uint32_t num_packet_headers_storable = get_compile_time_arg_val(2);
constexpr BufferType buffer0_type = static_cast<BufferType>(get_compile_time_arg_val(3));
constexpr uint32_t cb0_id = get_compile_time_arg_val(4);
constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(5);
constexpr uint32_t tensor0_page_size = get_compile_time_arg_val(6);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(7);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(8);
constexpr bool last_dim = get_compile_time_arg_val(9);
constexpr uint32_t num_banks = get_compile_time_arg_val(10);
constexpr uint32_t bf8_dim3_type = get_compile_time_arg_val(11);

template <bool DRAM>
inline void fabric_send_contig_tiles_dim3_bf16(
    uint32_t num_tiles,
    uint32_t ring_size,
    uint32_t tile_cols_per_chip,
    InterleavedAddrGenFast<DRAM>& addrgen,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
    FabricConnectionManager& fabric_connection) {
    uint32_t total = 0;
    uint32_t tile_id = 0;
    uint32_t total_cols = tile_cols_per_chip * ring_size;
    uint32_t end_abs_tile_id = dim3_rel2abs_tile_id(num_tiles - 1, tile_cols_per_chip, ring_size, my_chip_id);
    while (total < num_tiles) {
        uint32_t abs_tile_id = dim3_rel2abs_tile_id(tile_id, tile_cols_per_chip, ring_size, my_chip_id);
        uint64_t noc0_dest_noc_addr = get_noc_addr(abs_tile_id, addrgen, 0 /*offset*/, 0 /*noc_id*/);
        size_t l1_read_addr = get_read_ptr(cb0_id);

        // check previous is sent: true -> current tiles was sent already as contiguous
        if (dim3_was_stride_sent(abs_tile_id, total_cols, tile_cols_per_chip, num_banks, my_chip_id)) {
            // check prev-prev is sent: true -> current tiles is not sent yet
            if (dim3_was_stride_sent(
                    abs_tile_id, total_cols, tile_cols_per_chip, packet_size_in_pages * num_banks, my_chip_id)) {
                cb_wait_front(cb0_id, packet_size_in_pages);
                write_and_advance_local_read_address_for_fabric_write(
                    noc0_dest_noc_addr,
                    pkt_hdr_forward,
                    pkt_hdr_backward,
                    fabric_connection,
                    l1_read_addr,
                    tensor0_page_size);
                tile_id++;
                total++;
                noc_async_writes_flushed();
                cb_pop_front(cb0_id, packet_size_in_pages);
            } else {
                tile_id++;  // skip tile as it is already processed
            }
        } else {
            cb_wait_front(cb0_id, packet_size_in_pages);
            // check whether there is contiguous tile, the tile is in the local chip/buffer
            if ((abs_tile_id + num_banks) <= end_abs_tile_id &&
                dim3_is_tile_in_local(abs_tile_id + num_banks, total_cols, tile_cols_per_chip, my_chip_id)) {
                write_and_advance_local_read_address_for_fabric_write(
                    noc0_dest_noc_addr,
                    pkt_hdr_forward,
                    pkt_hdr_backward,
                    fabric_connection,
                    l1_read_addr,
                    packet_size_in_pages * tensor0_page_size);
                tile_id++;
                total += packet_size_in_pages;
            } else {
                write_and_advance_local_read_address_for_fabric_write(
                    noc0_dest_noc_addr,
                    pkt_hdr_forward,
                    pkt_hdr_backward,
                    fabric_connection,
                    l1_read_addr,
                    tensor0_page_size);
                tile_id++;
                total++;
            }
            noc_async_writes_flushed();
            cb_pop_front(cb0_id, packet_size_in_pages);
        }
    }
}

template <bool DRAM>
inline void fabric_send_full_contig(
    uint32_t contig_total,
    uint32_t& tile_id,
    InterleavedAddrGenFast<DRAM>& addrgen,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
    FabricConnectionManager& fabric_connection) {
    uint32_t total_local = 0;
    while (total_local < contig_total) {
        cb_wait_front(cb0_id, packet_size_in_pages);
        size_t l1_read_addr = get_read_ptr(cb0_id);
        uint64_t noc0_dest_noc_addr = get_noc_addr(tile_id, addrgen, 0 /*offset*/, 0 /*noc_id*/);
        write_and_advance_local_read_address_for_fabric_write(
            noc0_dest_noc_addr,
            pkt_hdr_forward,
            pkt_hdr_backward,
            fabric_connection,
            l1_read_addr,
            packet_size_in_pages * tensor0_page_size);
        noc_async_writes_flushed();
        cb_pop_front(cb0_id, packet_size_in_pages);
        tile_id++;
        total_local++;
        if (total_local % num_banks == 0) {
            tile_id += num_banks * (packet_size_in_pages - 1);
        }
    }
}

template <bool DRAM>
inline void fabric_send_2contig_bf8(
    uint32_t contig_total,
    uint32_t& tile_id,
    InterleavedAddrGenFast<DRAM>& addrgen,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
    FabricConnectionManager& fabric_connection) {
    uint32_t total_local = 0;
    while (total_local < contig_total) {
        cb_wait_front(cb0_id, 2);
        size_t l1_read_addr = get_read_ptr(cb0_id);
        uint64_t noc0_dest_noc_addr = get_noc_addr(tile_id, addrgen, 0 /*offset*/, 0 /*noc_id*/);
        write_and_advance_local_read_address_for_fabric_write(
            noc0_dest_noc_addr,
            pkt_hdr_forward,
            pkt_hdr_backward,
            fabric_connection,
            l1_read_addr,
            2 * tensor0_page_size);
        tile_id++;
        total_local++;
        noc_async_writes_flushed();
        cb_pop_front(cb0_id, 2);
        if (total_local % num_banks == 0) {
            tile_id += num_banks;
        }
    }
}

template <bool DRAM>
inline void fabric_send_non_contig(
    uint32_t num_tiles,
    uint32_t& tile_id,
    InterleavedAddrGenFast<DRAM>& addrgen,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
    FabricConnectionManager& fabric_connection) {
    uint32_t total_local = 0;
    while (total_local < num_tiles) {
        uint32_t tiles_in_packet = std::min(num_tiles - total_local, packet_size_in_pages);
        cb_wait_front(cb0_id, tiles_in_packet);
        size_t l1_read_addr = get_read_ptr(cb0_id);
        for (uint32_t i = 0; i < tiles_in_packet; i++) {
            uint64_t noc0_dest_noc_addr = get_noc_addr(tile_id, addrgen, 0 /*offset*/, 0 /*noc_id*/);
            write_and_advance_local_read_address_for_fabric_write(
                noc0_dest_noc_addr,
                pkt_hdr_forward,
                pkt_hdr_backward,
                fabric_connection,
                l1_read_addr,
                tensor0_page_size);
            tile_id++;
        }
        noc_async_writes_flushed();
        cb_pop_front(cb0_id, tiles_in_packet);
        total_local += tiles_in_packet;
    }
}

template <bool DRAM>
inline void fabric_send_dim3_bf16_rest16_optimized(
    uint32_t num_tiles,
    uint32_t ring_size,
    uint32_t tile_cols_per_chip,
    InterleavedAddrGenFast<DRAM>& addrgen,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
    FabricConnectionManager& fabric_connection) {
    const uint32_t num_contig2 = 4;
    uint32_t row = num_tiles / tile_cols_per_chip;
    const uint32_t input_width = 16;
    uint32_t tile_id = input_width * my_chip_id;
    for (uint32_t i = 0; i < row; i++) {
        fabric_send_full_contig(num_contig2, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
        fabric_send_non_contig(8, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
        tile_id += input_width * (ring_size - 1) + 4;
    }
}

template <bool DRAM>
inline void fabric_send_dim3_bf16_rest8_optimized(
    uint32_t num_tiles,
    uint32_t ring_size,
    uint32_t tile_cols_per_chip,
    InterleavedAddrGenFast<DRAM>& addrgen,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
    FabricConnectionManager& fabric_connection) {
    const uint32_t num_contig2 = num_banks * 5;
    uint32_t row = num_tiles / tile_cols_per_chip;
    const uint32_t input_width = 128;
    uint32_t tile_id = input_width * my_chip_id;
    if constexpr (my_chip_id % 3 == 0) {
        for (uint32_t i = 0; i < row; i++) {
            if (i % 3 == 0) {
                fabric_send_full_contig(
                    num_contig2, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                fabric_send_non_contig(8, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
            } else if (i % 3 == 1) {
                fabric_send_non_contig(8, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                fabric_send_full_contig(
                    num_contig2, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
            } else {
                fabric_send_non_contig(4, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                fabric_send_full_contig(
                    num_contig2, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                fabric_send_non_contig(4, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
            }
            tile_id += input_width * (ring_size - 1);
        }
    } else if constexpr (my_chip_id % 3 == 1) {
        for (uint32_t i = 0; i < row; i++) {
            if (i % 3 == 0) {
                fabric_send_non_contig(4, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                fabric_send_full_contig(
                    num_contig2, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                fabric_send_non_contig(4, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
            } else if (i % 3 == 1) {
                fabric_send_full_contig(
                    num_contig2, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                fabric_send_non_contig(8, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
            } else {
                fabric_send_non_contig(8, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                fabric_send_full_contig(
                    num_contig2, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
            }
            tile_id += input_width * (ring_size - 1);
        }
    } else {
        for (uint32_t i = 0; i < row; i++) {
            if (i % 3 == 0) {
                fabric_send_non_contig(8, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                fabric_send_full_contig(
                    num_contig2, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
            } else if (i % 3 == 1) {
                fabric_send_non_contig(4, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                fabric_send_full_contig(
                    num_contig2, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                fabric_send_non_contig(4, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
            } else {
                fabric_send_full_contig(
                    num_contig2, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                fabric_send_non_contig(8, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
            }
            tile_id += input_width * (ring_size - 1);
        }
    }
}

template <bool DRAM>
inline void fabric_send_llama_8b_n300(
    uint32_t num_tiles,
    uint32_t ring_size,
    uint32_t tile_cols_per_chip,
    InterleavedAddrGenFast<DRAM>& addrgen,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
    FabricConnectionManager& fabric_connection) {
    static const uint32_t input_width = 2004;
    constexpr uint32_t num_full_contig = 41 * 12;
    constexpr uint32_t num_2contig = 12;
    constexpr uint32_t rest_tiles = 12;
    uint32_t tile_id = input_width * my_chip_id;
    fabric_send_full_contig(num_full_contig, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
    fabric_send_2contig_bf8(num_2contig, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
    fabric_send_non_contig(rest_tiles, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
}

template <bool DRAM>
inline void fabric_send_falcon40(
    uint32_t num_tiles,
    uint32_t ring_size,
    uint32_t tile_cols_per_chip,
    InterleavedAddrGenFast<DRAM>& addrgen,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
    FabricConnectionManager& fabric_connection) {
    const uint32_t num_contig2 = num_banks;
    uint32_t row = num_tiles / tile_cols_per_chip;
    if constexpr ((BF8_DIM3_TYPE)bf8_dim3_type == T3K_FALCON40_8192) {
        const uint32_t input_width = 32;  // 8192/8/32
        uint32_t tile_id = input_width * my_chip_id;
        if constexpr (my_chip_id % 3 == 0) {
            for (uint32_t i = 0; i < row; i++) {
                if (i % 3 == 0) {
                    fabric_send_2contig_bf8(
                        num_contig2, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                    fabric_send_non_contig(8, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                } else if (i % 3 == 1) {
                    fabric_send_non_contig(8, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                    fabric_send_2contig_bf8(
                        num_contig2, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                } else {
                    fabric_send_non_contig(4, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                    fabric_send_2contig_bf8(
                        num_contig2, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                    fabric_send_non_contig(4, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                }
                tile_id += input_width * (ring_size - 1);
            }
        } else if constexpr (my_chip_id % 3 == 1) {
            for (uint32_t i = 0; i < row; i++) {
                if (i % 3 == 0) {
                    fabric_send_non_contig(4, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                    fabric_send_2contig_bf8(
                        num_contig2, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                    fabric_send_non_contig(4, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                } else if (i % 3 == 1) {
                    fabric_send_2contig_bf8(
                        num_contig2, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                    fabric_send_non_contig(8, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                } else {
                    fabric_send_non_contig(8, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                    fabric_send_2contig_bf8(
                        num_contig2, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                }
                tile_id += input_width * (ring_size - 1);
            }
        } else {
            for (uint32_t i = 0; i < row; i++) {
                if (i % 3 == 0) {
                    fabric_send_non_contig(8, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                    fabric_send_2contig_bf8(
                        num_contig2, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                } else if (i % 3 == 1) {
                    fabric_send_non_contig(4, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                    fabric_send_2contig_bf8(
                        num_contig2, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                    fabric_send_non_contig(4, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                } else {
                    fabric_send_2contig_bf8(
                        num_contig2, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                    fabric_send_non_contig(8, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                }
                tile_id += input_width * (ring_size - 1);
            }
        }
    } else if constexpr ((BF8_DIM3_TYPE)bf8_dim3_type == T3K_FALCON40_32768) {
        const uint32_t input_width = 128;  // 32768/8/32
        uint32_t tile_id = input_width * my_chip_id;
        uint32_t num_full_contig = 24;
        if constexpr (my_chip_id % 3 == 0) {
            for (uint32_t i = 0; i < row; i++) {
                if (i % 3 == 0) {
                    fabric_send_full_contig(
                        num_full_contig, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                    fabric_send_2contig_bf8(
                        num_contig2, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                    fabric_send_non_contig(8, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                } else if (i % 3 == 1) {
                    fabric_send_non_contig(8, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                    fabric_send_full_contig(
                        num_full_contig, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                    fabric_send_2contig_bf8(
                        num_contig2, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                } else {
                    fabric_send_non_contig(4, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                    fabric_send_full_contig(
                        num_full_contig, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                    fabric_send_2contig_bf8(
                        num_contig2, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                    fabric_send_non_contig(4, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                }
                tile_id += input_width * (ring_size - 1);
            }
        } else if constexpr (my_chip_id % 3 == 1) {
            for (uint32_t i = 0; i < row; i++) {
                if (i % 3 == 0) {
                    fabric_send_non_contig(4, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                    fabric_send_full_contig(
                        num_full_contig, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                    fabric_send_2contig_bf8(
                        num_contig2, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                    fabric_send_non_contig(4, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                } else if (i % 3 == 1) {
                    fabric_send_full_contig(
                        num_full_contig, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                    fabric_send_2contig_bf8(
                        num_contig2, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                    fabric_send_non_contig(8, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                } else {
                    fabric_send_non_contig(8, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                    fabric_send_full_contig(
                        num_full_contig, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                    fabric_send_2contig_bf8(
                        num_contig2, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                }
                tile_id += input_width * (ring_size - 1);
            }
        } else {
            for (uint32_t i = 0; i < row; i++) {
                if (i % 3 == 0) {
                    fabric_send_non_contig(8, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                    fabric_send_full_contig(
                        num_full_contig, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                    fabric_send_2contig_bf8(
                        num_contig2, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                } else if (i % 3 == 1) {
                    fabric_send_non_contig(4, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                    fabric_send_full_contig(
                        num_full_contig, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                    fabric_send_2contig_bf8(
                        num_contig2, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                    fabric_send_non_contig(4, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                } else {
                    fabric_send_full_contig(
                        num_full_contig, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                    fabric_send_2contig_bf8(
                        num_contig2, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                    fabric_send_non_contig(8, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
                }
                tile_id += input_width * (ring_size - 1);
            }
        }
    }
}

template <bool DRAM>
inline void fabric_send_dim2_bf8(
    uint32_t filled_bank_tiles,
    uint32_t rest_full_contig_ids,
    uint32_t tile_id_start,
    uint32_t& total,
    uint32_t rest_tiles,
    uint32_t ring_size,
    uint32_t tile_cols_per_chip,
    InterleavedAddrGenFast<DRAM>& tensor0_addrgen,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
    FabricConnectionManager& fabric_connection) {
    // e.g. num_banks: 4, packet_size_in_pages: 4
    //      |       chip0           |
    //    0 | id 0|    1|    2|    3|
    //      |    4|    5|    6|    7|
    //      |    8|    9|   10|   11|
    //    __|   12|   13|   14|   15|___  <- filled_bank_rows: 1, filled_bank_tiles: 1*4*4=16
    //    1 |   16|   17|   18|   19|     <- rest_tiles: 14 (16-29)
    //      |   20|   21|   22|   23|     <- rest_full_contig_ids: 2 (16, 17)
    //      |   24|   25|   26|   27|     <- rest_half_contig_ids: 2 (18, 19)
    //    __|   28|   29|            __   <- rest_orphan_tiles: 1 (26, 27)

    uint32_t rest_half_contig_ids, rest_orphan_tiles;
    if (num_banks * 3 < rest_tiles) {
        rest_half_contig_ids = (num_banks - rest_full_contig_ids);
        rest_orphan_tiles = rest_half_contig_ids;
    } else if (num_banks * 2 <= rest_tiles) {
        rest_half_contig_ids = num_banks;
        rest_orphan_tiles = (rest_tiles) % (num_banks * 2);
    } else if (num_banks < rest_tiles) {
        rest_half_contig_ids = (rest_tiles) % num_banks;
        rest_orphan_tiles = num_banks - rest_half_contig_ids;
    } else {
        rest_half_contig_ids = 0;
        rest_orphan_tiles = rest_tiles;
    }
    uint32_t num_tiles = rest_half_contig_ids + (filled_bank_tiles + rest_full_contig_ids);
    uint32_t outer_id = 0;

    // send half (2) contig tiles twice in one loop
    while (total < num_tiles) {
        uint32_t num_2contig = min(rest_half_contig_ids - outer_id, 2);
        cb_wait_front(cb0_id, packet_size_in_pages);
        size_t l1_read_addr = get_read_ptr(cb0_id);

        uint32_t id = total + tile_id_start;
        for (uint32_t j = 0; j < num_2contig; j++) {
            uint64_t noc0_dest_noc_addr = get_noc_addr(id, tensor0_addrgen, 0 /*offset*/, 0 /*noc_id*/);
            write_and_advance_local_read_address_for_fabric_write(
                noc0_dest_noc_addr,
                pkt_hdr_forward,
                pkt_hdr_backward,
                fabric_connection,
                l1_read_addr,
                2 * tensor0_page_size);
            id++;
        }
        outer_id += num_2contig;
        total += num_2contig;
        if (total % num_banks == 0) {
            total += num_banks + rest_full_contig_ids;
        }
        noc_async_writes_flushed();
        cb_pop_front(cb0_id, packet_size_in_pages);
    }
    total += tile_id_start;
    fabric_send_non_contig(
        rest_orphan_tiles, total, tensor0_addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
}

template <bool DRAM>
inline void fabric_send_dim2_bf16(
    uint32_t num_tiles_per_chip,
    uint32_t& total,
    uint32_t rest_tiles,
    uint32_t ring_size,
    uint32_t tile_cols_per_chip,
    InterleavedAddrGenFast<DRAM>& tensor0_addrgen,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
    FabricConnectionManager& fabric_connection) {
    auto rest_orphan_tiles = 0;
    if (num_banks * 1 < rest_tiles) {
        rest_orphan_tiles = num_banks - (rest_tiles % (num_banks * (packet_size_in_pages - 1)));
    } else {
        rest_orphan_tiles = num_tiles_per_chip % (num_banks * packet_size_in_pages);
    }

    fabric_send_non_contig(
        rest_orphan_tiles, total, tensor0_addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
}

template <bool DRAM>
inline void fabric_send_dim2(
    uint32_t num_tiles_per_chip,
    uint32_t tile_id_start,
    uint32_t ring_size,
    uint32_t tile_cols_per_chip,
    InterleavedAddrGenFast<DRAM>& tensor0_addrgen,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
    FabricConnectionManager& fabric_connection) {
    // e.g. num_banks: 4, packet_size_in_pages: 2
    //      |       chip0           |___
    //    0 | id 0|    1|    2|    3|
    //    __|    4|    5|    6|    7|___
    //    1 |    8|    9|   10|   11|
    //    __|   12|   13|   14|   15|___  <- filled_bank_rows: 2, filled_bank_tiles: 2*4*2=16
    //    2 |   16|   17|   18|   19|     <- rest_tiles: 7 (16-22), rest_orphan_tiles: 1 (19)
    //    __|   20|   21|   22|           <- rest_full_contig_ids: 3 (16,17,18,20,21,22)
    //     ---------------------------------------------------
    //      |       chip1           |
    //      |   23|  24| ......

    auto filled_bank_rows = num_tiles_per_chip / (num_banks * packet_size_in_pages);
    auto rest_tiles = num_tiles_per_chip % (num_banks * packet_size_in_pages);
    auto filled_bank_tiles = filled_bank_rows * num_banks * packet_size_in_pages;
    auto rest_full_contig_ids = 0;
    auto rest_full_contig_rows = 0;
    uint32_t total = 0;
    if (num_banks * (packet_size_in_pages - 1) < rest_tiles) {
        rest_full_contig_ids = (rest_tiles) % (num_banks * (packet_size_in_pages - 1));
    }
    // send fully contig tiles. e.g. tileID: 0-15, 16-18, 20-22
    total += tile_id_start;
    fabric_send_full_contig(
        filled_bank_rows * num_banks + rest_full_contig_ids,
        total,
        tensor0_addrgen,
        pkt_hdr_forward,
        pkt_hdr_backward,
        fabric_connection);
    total -= tile_id_start;

    if constexpr (packet_size_in_pages == 2) {  // bf16
        // e.g. tileID: 19
        total += tile_id_start;
        fabric_send_dim2_bf16(
            num_tiles_per_chip,
            total,
            rest_tiles,
            ring_size,
            tile_cols_per_chip,
            tensor0_addrgen,
            pkt_hdr_forward,
            pkt_hdr_backward,
            fabric_connection);
    } else {  // bf8
        fabric_send_dim2_bf8(
            filled_bank_tiles,
            rest_full_contig_ids,
            tile_id_start,
            total,
            rest_tiles,
            ring_size,
            tile_cols_per_chip,
            tensor0_addrgen,
            pkt_hdr_forward,
            pkt_hdr_backward,
            fabric_connection);
    }
}

/*
 * CCL Send will present various operating modes. Although there is only a single send kernel, it may (compile time)
 * dispatch implementations depending on those invocation parameters.
 */
void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    size_t arg_idx = 0;
    // Load the input tensor spec
    address_t tensor_address0 = get_arg_val<address_t>(arg_idx++);
    const size_t out_ready_sem_bank_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t tile_id_start = get_arg_val<uint32_t>(arg_idx++);
    uint32_t tile_id_end = get_arg_val<uint32_t>(arg_idx++);
    bool wait_output_semaphore = get_arg_val<uint32_t>(arg_idx++);
    bool reset_global_semaphore = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_ready_sem_wait_value = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_tiles_per_chip = get_arg_val<uint32_t>(arg_idx++);
    uint32_t ring_size = get_arg_val<uint32_t>(arg_idx++);
    uint32_t tile_cols_per_chip = get_arg_val<uint32_t>(arg_idx++);
    size_t arg_for_fab = arg_idx;
    auto fabric_connection = FabricConnectionManager::build_from_args(arg_idx);

    DPRINT << "ct args: \n";
    DPRINT << "my_chip_id: " << (uint32_t)my_chip_id << "\n";
    DPRINT << "reserved_packet_header_cb_id: " << (uint32_t)reserved_packet_header_cb_id << "\n";
    DPRINT << "num_packet_headers_storable: " << (uint32_t)num_packet_headers_storable << "\n";
    DPRINT << "buffer0_type: " << (uint32_t)buffer0_type << "\n";
    DPRINT << "cb0_id: " << (uint32_t)cb0_id << "\n";
    DPRINT << "packet_size_in_pages: " << (uint32_t)packet_size_in_pages << "\n";
    DPRINT << "tensor0_page_size: " << (uint32_t)tensor0_page_size << "\n";
    DPRINT << "num_targets_forward_direction: " << (uint32_t)num_targets_forward_direction << "\n";
    DPRINT << "num_targets_backward_direction: " << (uint32_t)num_targets_backward_direction << "\n";
    DPRINT << "last_dim: " << (uint32_t)last_dim << "\n";
    DPRINT << "num_banks: " << (uint32_t)num_banks << "\n";
    DPRINT << "bf8_dim3_type: " << (uint32_t)bf8_dim3_type << "\n";

    DPRINT << "rt args: \n";
    DPRINT << "tensor_address0: " << (uint32_t)tensor_address0 << "\n";
    DPRINT << "tile_id_start: " << (uint32_t)tile_id_start << "\n";
    DPRINT << "tile_id_end: " << (uint32_t)tile_id_end << "\n";
    DPRINT << "wait_output_semaphore: " << (uint32_t)wait_output_semaphore << "\n";
    DPRINT << "reset_global_semaphore: " << (uint32_t)reset_global_semaphore << "\n";
    DPRINT << "out_ready_sem_bank_addr: " << (uint32_t)out_ready_sem_bank_addr << "\n";
    DPRINT << "out_ready_sem_noc0_x: " << (uint32_t)out_ready_sem_noc0_x << "\n";
    DPRINT << "out_ready_sem_noc0_y: " << (uint32_t)out_ready_sem_noc0_y << "\n";
    DPRINT << "out_ready_sem_wait_value: " << (uint32_t)out_ready_sem_wait_value << "\n";
    DPRINT << "num_tiles_per_chip: " << (uint32_t)num_tiles_per_chip << "\n";
    DPRINT << "ring_size: " << (uint32_t)ring_size << "\n";
    DPRINT << "tile_cols_per_chip: " << (uint32_t)tile_cols_per_chip << "\n";

    DPRINT << "arg_for_fab: " << (uint32_t)arg_for_fab << "\n";
    DPRINT << "fabric_connection arg 0" << get_arg_val<uint32_t>(arg_for_fab++) << "\n";
    DPRINT << "fabric_connection arg 1" << get_arg_val<uint32_t>(arg_for_fab++) << "\n";
    DPRINT << "fabric_connection arg 2" << get_arg_val<uint32_t>(arg_for_fab++) << "\n";
    DPRINT << "fabric_connection arg 3" << get_arg_val<uint32_t>(arg_for_fab++) << "\n";
    DPRINT << "fabric_connection arg 4" << get_arg_val<uint32_t>(arg_for_fab++) << "\n";

    // packet header cb
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_addr_forward = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_addr_backward = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_seminc = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);
    DPRINT << "packet_header_buffer_addr_forward: " << (uint32_t)packet_header_buffer_addr_forward << "\n";
    DPRINT << "packet_header_buffer_addr_backward: " << (uint32_t)packet_header_buffer_addr_backward << "\n";
    DPRINT << "packet_header_buffer_seminc: " << (uint32_t)packet_header_buffer_seminc << "\n";

    // pre-populate packet headers
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_forward);
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_backward);
    pkt_hdr_forward->to_chip_multicast(
        tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_forward_direction)});
    pkt_hdr_backward->to_chip_multicast(
        tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_backward_direction)});

    // interleaved addrgen
    constexpr bool is_dram = buffer0_type == tt::tt_metal::BufferType::DRAM;
    auto tensor0_addrgen = InterleavedAddrGenFast<is_dram>{
        .bank_base_address = tensor_address0, .page_size = tensor0_page_size, .data_format = get_dataformat(cb0_id)};

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.open();
    }

    // 1. mcast via fabric to remote tensor addresses
    DPRINT << "num_targets_forward_direction: " << num_targets_forward_direction << "\n";
    DPRINT << "num_targets_backward_direction: " << num_targets_backward_direction << "\n";
    DPRINT << "my_chip_id: " << my_chip_id << "\n";

    DPRINT << "tensor -> CB: " << (uint32_t)cb0_id << "\n";
    DPRINT << "packet size in pages: " << (uint32_t)packet_size_in_pages << "\n";

    // when last_dim == true, tile_id coordinate is as follows
    //      |        chip0          |       chip1           |
    //      |  tile_cols_per_chip   |                       |
    //      | id 0|    1|    2|    3|    4|    5|    6|    7|
    // row  |    8|    9|   10|   11|   12|   13|   14|   15|
    //      |   16|   17|   18|   19|   20|   21|   22|   23|
    //      |   24|   25|   26|   27|   28|   29|   30|   31|
    //
    // else (dim == 1 or dim == 2)
    //      |                     chip0                     |
    //      | id 0|    1|    2|    3|    4|    5|    6|    7|
    //      |    8|    9|   10|   11|   12|   13|   14|   15|
    //     ---------------------------------------------------
    //      |                     chip1                     |
    //      |   16|   17|   18|   19|   20|   21|   22|   23|
    //      |   24|   25|   26|   27|   28|   29|   30|   31|
    //

    if constexpr (last_dim) {
        if constexpr (packet_size_in_pages == 2) {  // bf16
            if (is_dram && tile_cols_per_chip == 128) {
                fabric_send_dim3_bf16_rest8_optimized<is_dram>(
                    num_tiles_per_chip,
                    ring_size,
                    tile_cols_per_chip,
                    tensor0_addrgen,
                    pkt_hdr_forward,
                    pkt_hdr_backward,
                    fabric_connection);
            } else if (is_dram && tile_cols_per_chip == 16) {
                fabric_send_dim3_bf16_rest16_optimized<is_dram>(
                    num_tiles_per_chip,
                    ring_size,
                    tile_cols_per_chip,
                    tensor0_addrgen,
                    pkt_hdr_forward,
                    pkt_hdr_backward,
                    fabric_connection);
            } else {
                fabric_send_contig_tiles_dim3_bf16<is_dram>(
                    num_tiles_per_chip,
                    ring_size,
                    tile_cols_per_chip,
                    tensor0_addrgen,
                    pkt_hdr_forward,
                    pkt_hdr_backward,
                    fabric_connection);
            }
        } else {
            if constexpr (
                (BF8_DIM3_TYPE)bf8_dim3_type == T3K_FALCON40_8192 ||
                (BF8_DIM3_TYPE)bf8_dim3_type == T3K_FALCON40_32768) {
                fabric_send_falcon40<is_dram>(
                    num_tiles_per_chip,
                    ring_size,
                    tile_cols_per_chip,
                    tensor0_addrgen,
                    pkt_hdr_forward,
                    pkt_hdr_backward,
                    fabric_connection);
            } else if constexpr ((BF8_DIM3_TYPE)bf8_dim3_type == LLAMA_8B_N300) {
                fabric_send_llama_8b_n300<is_dram>(
                    num_tiles_per_chip,
                    ring_size,
                    tile_cols_per_chip,
                    tensor0_addrgen,
                    pkt_hdr_forward,
                    pkt_hdr_backward,
                    fabric_connection);
            }
        }
    } else {
        fabric_send_dim2(
            num_tiles_per_chip,
            tile_id_start,
            ring_size,
            tile_cols_per_chip,
            tensor0_addrgen,
            pkt_hdr_forward,
            pkt_hdr_backward,
            fabric_connection);
    }

    // 2. mcast output ready semaphore
    uint64_t out_ready_sem_noc_addr_in_pkt =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bank_addr, 0);
    auto* pkt_hdr = reinterpret_cast<PACKET_HEADER_TYPE*>(packet_header_buffer_seminc);
    pkt_hdr->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
        out_ready_sem_noc_addr_in_pkt,
        static_cast<uint16_t>(1),  // increment 1
        32});
    // Write the mcast packet (forward)
    if (fabric_connection.has_forward_connection()) {
        fabric_connection.get_forward_connection().wait_for_empty_write_slot();
        pkt_hdr->to_chip_multicast(
            tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_forward_direction)});
        fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
            packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
    }
    // Write the mcast packet (backward)
    if (fabric_connection.has_backward_connection()) {
        pkt_hdr->to_chip_multicast(
            tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_backward_direction)});
        fabric_connection.get_backward_connection().wait_for_empty_write_slot();
        fabric_connection.get_backward_connection().send_payload_non_blocking_from_address(
            packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
    }
    // increment locally
    uint64_t out_ready_sem_noc_addr =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bank_addr);
    noc_semaphore_inc(out_ready_sem_noc_addr, 1);

    // 3. wait for mcast output ready semaphore
    if (wait_output_semaphore) {
        while (*reinterpret_cast<volatile uint32_t*>(out_ready_sem_bank_addr) < out_ready_sem_wait_value);
    }

    // 4. global semaphore reset
    if (reset_global_semaphore) {
        const uint64_t dest_noc_addr = get_noc_addr(my_x[0], my_y[0], out_ready_sem_bank_addr);
        noc_inline_dw_write(dest_noc_addr, 0);
    }

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close();
    }

    noc_async_write_barrier();
    DPRINT << "DONE \n";
}
