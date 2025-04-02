// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_constants.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/noc_addr.hpp"
#include <tt-metalium/fabric_edm_packet_header.hpp>
#include <cstdint>
#include <utility>

enum BF8_DIM3_TYPE { NONE, BF8_DIM3_DRAM_REMAINDER_0, BF8_DIM3_REMAINDER_32 };

FORCE_INLINE void write_and_advance_local_read_address_for_fabric_write(
    uint64_t noc0_dest_noc_addr,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
    FabricConnectionManager& fabric_connection,
    size_t& l1_read_addr,
    uint32_t payload_size_bytes) {
    const auto [dest_noc_xy, dest_addr] = get_noc_address_components(noc0_dest_noc_addr);
    const size_t payload_l1_address = l1_read_addr;

    pkt_hdr_forward->to_noc_unicast_write(
        tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_noc_addr}, payload_size_bytes);
    pkt_hdr_backward->to_noc_unicast_write(
        tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_noc_addr}, payload_size_bytes);

    noc_async_write(payload_l1_address, safe_get_noc_addr(dest_noc_xy.x, dest_noc_xy.y, dest_addr), payload_size_bytes);
    if (fabric_connection.has_forward_connection()) {
        fabric_connection.get_forward_connection().wait_for_empty_write_slot();
        fabric_connection.get_forward_connection().send_payload_without_header_non_blocking_from_address(
            l1_read_addr, payload_size_bytes);
        fabric_connection.get_forward_connection().send_payload_flush_non_blocking_from_address(
            (uint32_t)pkt_hdr_forward, sizeof(PACKET_HEADER_TYPE));
    }

    if (fabric_connection.has_backward_connection()) {
        fabric_connection.get_backward_connection().wait_for_empty_write_slot();
        fabric_connection.get_backward_connection().send_payload_without_header_non_blocking_from_address(
            l1_read_addr, payload_size_bytes);
        fabric_connection.get_backward_connection().send_payload_flush_non_blocking_from_address(
            (uint32_t)pkt_hdr_backward, sizeof(PACKET_HEADER_TYPE));
    }

    noc_async_writes_flushed();

    l1_read_addr += payload_size_bytes;
}

inline bool dim3_is_tile_in_local(
    uint32_t tile_id, uint32_t total_cols, uint32_t tile_cols_per_chip, uint32_t my_chip_id) {
    return (uint32_t)my_chip_id == (tile_id % total_cols) / tile_cols_per_chip;
}

inline bool dim3_was_stride_sent(
    uint32_t tile_id, uint32_t total_cols, uint32_t tile_cols_per_chip, uint32_t stride, uint32_t my_chip_id) {
    return (tile_id >= stride) && dim3_is_tile_in_local(tile_id - stride, total_cols, tile_cols_per_chip, my_chip_id);
}

// convert tile id from input based to output based
inline uint32_t dim3_rel2abs_tile_id(
    uint32_t rel_tile_id, uint32_t tile_cols_per_chip, uint32_t ring_size, uint32_t ring_idx) {
    uint32_t row = rel_tile_id / tile_cols_per_chip;
    uint32_t idx = rel_tile_id % tile_cols_per_chip;
    return idx + (tile_cols_per_chip * ring_idx) + row * ring_size * tile_cols_per_chip;
}

// from output to input
inline uint32_t dim3_abs2rel_tile_id(
    uint32_t abs_tile_id, uint32_t tile_cols_per_chip, uint32_t ring_size, uint32_t ring_idx) {
    uint32_t row = abs_tile_id / (tile_cols_per_chip * ring_size);
    uint32_t id = (abs_tile_id % (tile_cols_per_chip * ring_size)) % tile_cols_per_chip;
    return tile_cols_per_chip * row + id;
}
