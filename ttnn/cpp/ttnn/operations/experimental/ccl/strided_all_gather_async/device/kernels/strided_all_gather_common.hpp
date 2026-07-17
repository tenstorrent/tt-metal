// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include <cstdint>
#include <utility>
#include "ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "subchunk_bands.hpp"

using namespace tt::tt_fabric::linear::experimental;

FORCE_INLINE uint32_t div_up(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

FORCE_INLINE uint32_t round_up(uint32_t a, uint32_t b) { return b * div_up(a, b); }

FORCE_INLINE void advance_subchunk(
    uint32_t& tensor_row, uint32_t& subchunk_start_row, uint32_t& subchunk_end_row, uint32_t subchunk_height_stride) {
    uint32_t new_subchunk_rows = tensor_row - subchunk_end_row;
    subchunk_start_row = subchunk_start_row + subchunk_height_stride;
    subchunk_end_row = subchunk_end_row + subchunk_height_stride;
    tensor_row = subchunk_start_row + (new_subchunk_rows - 1);
}

FORCE_INLINE int32_t get_chunk_tile(
    uint32_t& tensor_row,
    uint32_t& tensor_col,
    uint32_t num_workers,
    uint32_t& subchunk_start_row,
    uint32_t& subchunk_end_row,
    uint32_t subchunk_height_stride,
    uint32_t chunk_start_col,
    uint32_t chunk_end_col,
    uint32_t chunk_width,
    uint32_t tensor_Wt,
    uint32_t tensor_Ht) {
    // Compute the returned tile index
    uint32_t tile_index = -1;
    if (tensor_row < tensor_Ht) {
        tile_index = tensor_row * tensor_Wt + tensor_col;
    }

    // Update the next tensor row and col
    tensor_col = tensor_col + num_workers;
    if (tensor_col > chunk_end_col) {
        uint32_t tensor_col_chunk_space = tensor_col - chunk_start_col;
        uint32_t tensor_row_delta = tensor_col_chunk_space / chunk_width;
        tensor_row = tensor_row + tensor_row_delta;
        tensor_col = chunk_start_col + tensor_col_chunk_space - (tensor_row_delta * chunk_width);
        if (tensor_row > subchunk_end_row) {
            advance_subchunk(tensor_row, subchunk_start_row, subchunk_end_row, subchunk_height_stride);
        }
    }

    // Return the tile index
    return tile_index;
}

FORCE_INLINE uint32_t
get_sender_id(uint32_t direction, uint32_t my_chip_id, uint32_t slices_received, uint32_t ring_size) {
    int32_t sender_chip_id;
    if (direction == 1) {
        sender_chip_id = my_chip_id + slices_received + 1;
        return (sender_chip_id >= (int)ring_size) ? sender_chip_id - ring_size : sender_chip_id;
    } else {
        sender_chip_id = my_chip_id - (slices_received + 1);
        return (sender_chip_id < 0) ? ring_size + sender_chip_id : sender_chip_id;
    }
}

FORCE_INLINE uint32_t next_mm_aligned_chunk_height(
    uint32_t input_chunk_start_tile, uint32_t M_tiles_per_core, uint32_t input_tensor_Wt, uint32_t mm_block_h) {
    uint32_t input_row = input_chunk_start_tile / input_tensor_Wt;
    if ((input_row + mm_block_h) > M_tiles_per_core) {
        return M_tiles_per_core - input_row;
    } else {
        return mm_block_h;
    }
}

// Advance chunk_start_tile past one chunk without moving data. Mirrors the
// tile-advance tail of read_chunk/write_chunk so a direction that relays only
// half of a split slice lands on the same M-block boundary a full traversal would.
FORCE_INLINE void advance_chunk_start_tile(
    uint32_t& chunk_start_tile, uint32_t chunk_width, uint32_t subchunk_height, uint32_t input_tensor_Wt) {
    uint32_t chunk_start_row = chunk_start_tile / input_tensor_Wt;
    uint32_t new_chunk_start_tile = chunk_start_tile + chunk_width;
    uint32_t new_chunk_row = new_chunk_start_tile / input_tensor_Wt;
    if (new_chunk_row != chunk_start_row) {
        chunk_start_tile = (chunk_start_row + subchunk_height) * input_tensor_Wt;
    } else {
        chunk_start_tile = new_chunk_start_tile;
    }
}

template <typename AddrGenType>
FORCE_INLINE uint32_t read_chunk(
    uint32_t& chunk_start_tile,
    uint32_t worker_tile_offset,
    uint32_t cb_output_id,
    uint32_t tiles_in_chunk,
    uint32_t chunk_width,
    uint32_t subchunk_height,
    uint32_t subchunk_height_stride,
    uint32_t max_tiles_per_packet,
    uint32_t ag_worker_core_id,
    uint32_t ag_worker_cores,
    AddrGenType input_tensor_addrgen,
    uint32_t input_tensor_page_size,
    AddrGenType output_tensor_addrgen,
    uint32_t input_tensor_Wt,
    uint32_t input_tensor_Ht,
    uint32_t output_tensor_Wt,
    uint32_t actual_sender_chip_id,
    bool read_output,
    // Must mirror write_chunk's band decomposition exactly: this reader is the CB producer feeding
    // write_chunk (the consumer) on the same core, so the per-band tile order must match or the relay
    // maps source tiles to the wrong output addresses. mm_sub_chunks==1 => whole-chunk (baseline).
    uint32_t mm_cores_y = 1,
    uint32_t mm_sub_chunks = 1) {
    // Chunk values (chunk spans all mm cores)
    // convert chunk start from linear index into row and col coord, still in input tensor space
    uint32_t chunk_start_row = chunk_start_tile / input_tensor_Wt;
    uint32_t chunk_start_col_base = chunk_start_tile % input_tensor_Wt;
    uint32_t chunk_start_col =
        read_output ? (chunk_start_col_base + actual_sender_chip_id * input_tensor_Wt) : chunk_start_col_base;
    uint32_t chunk_end_col = chunk_start_col + chunk_width - 1;

    Noc noc_obj;
    CircularBuffer cb_output(cb_output_id);

    for (uint32_t band = 0; band < mm_sub_chunks; band++) {
        uint32_t band_lo, band_h;
        balanced_band(subchunk_height, mm_sub_chunks, band, band_lo, band_h);
        if (band_h == 0) {
            break;  // empty trailing band, only when mm_sub_chunks > subchunk_height
        }
        uint32_t tiles_in_band = chunk_width * band_h * mm_cores_y;
        uint32_t worker_tiles_in_band =
            (tiles_in_band / ag_worker_cores) + ((ag_worker_core_id < (tiles_in_band % ag_worker_cores)) ? 1 : 0);
        uint32_t num_tiles_per_packet = std::min(max_tiles_per_packet, worker_tiles_in_band);
        uint32_t packets_in_band = div_up(worker_tiles_in_band, num_tiles_per_packet);
        uint32_t band_tile_iter = 0;

        uint32_t band_start_row = chunk_start_row + band_lo;
        uint32_t subchunk_start_row = band_start_row;
        uint32_t subchunk_end_row = band_start_row + band_h - 1;
        uint32_t worker_chunk_row = band_start_row + (worker_tile_offset / chunk_width);
        uint32_t worker_chunk_col = chunk_start_col + (worker_tile_offset % chunk_width);
        if (worker_chunk_row > subchunk_end_row) {
            advance_subchunk(worker_chunk_row, subchunk_start_row, subchunk_end_row, subchunk_height_stride);
        }

        for (uint32_t packet_idx = 0; packet_idx < packets_in_band; packet_idx++) {
            uint32_t tiles_left_in_band = worker_tiles_in_band - band_tile_iter;
            uint32_t tiles_to_read_in_packet = std::min(tiles_left_in_band, num_tiles_per_packet);

            cb_output.reserve_back(max_tiles_per_packet);
            size_t l1_write_addr = cb_output.get_write_ptr();
            for (uint32_t j = 0; j < tiles_to_read_in_packet; ++j) {
                int32_t tile_id = get_chunk_tile(
                    worker_chunk_row,
                    worker_chunk_col,
                    ag_worker_cores,
                    subchunk_start_row,
                    subchunk_end_row,
                    subchunk_height_stride,
                    chunk_start_col,
                    chunk_end_col,
                    chunk_width,
                    read_output ? output_tensor_Wt : input_tensor_Wt,
                    input_tensor_Ht);
                if (tile_id >= 0) {
                    // Device 2.0 migration: legacy primitive retained, precomposed uint64_t address
                    // from get_noc_addr(tile_id, accessor).
                    uint64_t noc_read_addr =
                        get_noc_addr(tile_id, read_output ? output_tensor_addrgen : input_tensor_addrgen);
                    noc_async_read(noc_read_addr, l1_write_addr, input_tensor_page_size);

                    l1_write_addr += input_tensor_page_size;
                }
                band_tile_iter++;
            }

            noc_obj.async_read_barrier();
            cb_output.push_back(max_tiles_per_packet);
        }
    }

    uint32_t new_chunk_start_tile = chunk_start_tile + chunk_width;
    uint32_t new_chunk_row = new_chunk_start_tile / input_tensor_Wt;
    if (new_chunk_row != chunk_start_row) {
        chunk_start_tile = (chunk_start_row + subchunk_height) * input_tensor_Wt;
    } else {
        chunk_start_tile = new_chunk_start_tile;
    }
    return 0;  // return value is unused by callers
}

// FabricSenderType is deduced from mux_connection so the same body serves any fabric sender that
// satisfies the linear-API contract (WorkerToFabricMuxSender for Mux V1, FabricMuxV2Sender for V2).
template <typename AddrGenType, typename FabricSenderType>
FORCE_INLINE uint32_t write_chunk(
    uint32_t& chunk_start_tile,
    uint32_t worker_tile_offset,
    uint32_t cb_output_id,
    uint32_t tiles_in_chunk,
    uint32_t chunk_width,
    uint32_t subchunk_height,
    uint32_t subchunk_height_stride,
    uint32_t max_tiles_per_packet,
    uint32_t ag_worker_core_id,
    uint32_t ag_worker_cores,
    AddrGenType output_addrgen,
    uint32_t output_page_size,
    uint32_t input_tensor_Wt,
    uint32_t input_tensor_Ht,
    uint32_t output_tensor_Wt,
    uint32_t actual_sender_chip_id,
    FabricSenderType& mux_connection,
    volatile PACKET_HEADER_TYPE* pkt_scatter_hdr,
    volatile PACKET_HEADER_TYPE* pkt_unicast_hdr,
    volatile PACKET_HEADER_TYPE* pkt_hdr_sem_inc,
    uint64_t out_ready_sem_noc_addr_in_pkt,
    const bool direction,
    const uint32_t num_targets_forward_direction,
    const uint32_t num_targets_backward_direction,
    bool write_local,
    // Streaming matmul signal (Option W, M sub-chunking). The chunk's subchunk_height M-rows are
    // delivered as mm_sub_chunks row-bands: for each band the walk sweeps the band's rows of EVERY
    // injector (injector-major, same as the whole-chunk walk but height-limited), then fires one
    // matmul-aggregator inc. Because a band is a contiguous row-range of all injectors, "band s landed" is
    // true for every injector at once -- the invariant the injector reader relies on. The out_ready
    // sem and chunk_start advance stay per-chunk (the AG receiver counts one per chunk).
    // mm_sub_chunks==1 => a single band spanning the whole chunk => byte-identical to no sub-chunking.
    uint32_t mm_cores_y = 1,
    uint32_t mm_sub_chunks = 1,
    bool mm_signal_enabled = false,
    volatile PACKET_HEADER_TYPE* pkt_hdr_mm_sem_inc = nullptr,
    uint64_t mm_agg_sem_noc_addr = 0) {
    // Chunk values (chunk spans all mm cores)
    // convert chunk start from linear index into row and col coord, still in input tensor space
    uint32_t chunk_start_row = chunk_start_tile / input_tensor_Wt;
    uint32_t chunk_start_col_base = chunk_start_tile % input_tensor_Wt;
    uint32_t chunk_start_col = chunk_start_col_base + actual_sender_chip_id * input_tensor_Wt;
    uint32_t chunk_end_col = chunk_start_col + chunk_width - 1;

    Noc noc_obj;
    CircularBuffer cb_output(cb_output_id);

    for (uint32_t band = 0; band < mm_sub_chunks; band++) {
        uint32_t band_lo, band_h;
        balanced_band(subchunk_height, mm_sub_chunks, band, band_lo, band_h);
        if (band_h == 0) {
            break;  // empty trailing band, only when mm_sub_chunks > subchunk_height
        }
        uint32_t tiles_in_band = chunk_width * band_h * mm_cores_y;
        uint32_t worker_tiles_in_band =
            (tiles_in_band / ag_worker_cores) + ((ag_worker_core_id < (tiles_in_band % ag_worker_cores)) ? 1 : 0);
        uint32_t num_tiles_per_packet = std::min(max_tiles_per_packet, worker_tiles_in_band);
        uint32_t packets_in_band = div_up(worker_tiles_in_band, num_tiles_per_packet);
        uint32_t band_tile_iter = 0;

        // Subchunk tracker for this band: band_h rows of injector 0, starting band_lo rows into the
        // chunk; advance_subchunk jumps subchunk_height_stride (== M_tiles_per_core) to the next
        // injector's same band. Injector-major within the band.
        uint32_t band_start_row = chunk_start_row + band_lo;
        uint32_t subchunk_start_row = band_start_row;
        uint32_t subchunk_end_row = band_start_row + band_h - 1;

        uint32_t worker_chunk_row = band_start_row + (worker_tile_offset / chunk_width);
        uint32_t worker_chunk_col = chunk_start_col + (worker_tile_offset % chunk_width);
        if (worker_chunk_row > subchunk_end_row) {
            advance_subchunk(worker_chunk_row, subchunk_start_row, subchunk_end_row, subchunk_height_stride);
        }

        for (uint32_t packet_idx = 0; packet_idx < packets_in_band; packet_idx++) {
            uint32_t tiles_left_in_band = worker_tiles_in_band - band_tile_iter;
            uint32_t tiles_to_write_in_packet = std::min(tiles_left_in_band, num_tiles_per_packet);

            cb_output.wait_front(max_tiles_per_packet);
            size_t l1_read_addr = cb_output.get_read_ptr();

            // Collect the packet's valid dest tiles. get_chunk_tile advances the row/col cursor each
            // call and returns <0 for a padding hole; the reader front-packs the same valid tiles into
            // the source CB in this order, so source tile k maps to noc_addrs[k].
            uint64_t noc_addrs[NOC_SCATTER_WRITE_MAX_CHUNKS] = {0, 0, 0, 0};
            uint64_t local_noc_addrs[NOC_SCATTER_WRITE_MAX_CHUNKS] = {0, 0, 0, 0};
            uint16_t chunk_sizes[NOC_SCATTER_WRITE_MAX_CHUNKS - 1] = {
                static_cast<uint16_t>(output_page_size),
                static_cast<uint16_t>(output_page_size),
                static_cast<uint16_t>(output_page_size)};
            uint32_t valid_tiles = 0;
            for (uint32_t j = 0; j < tiles_to_write_in_packet; ++j) {
                int32_t tile_id = get_chunk_tile(
                    worker_chunk_row,
                    worker_chunk_col,
                    ag_worker_cores,
                    subchunk_start_row,
                    subchunk_end_row,
                    subchunk_height_stride,
                    chunk_start_col,
                    chunk_end_col,
                    chunk_width,
                    output_tensor_Wt,
                    input_tensor_Ht);
                if (tile_id >= 0) {
                    noc_addrs[valid_tiles] =
                        tt::tt_fabric::linear::addrgen_detail::get_noc_address(output_addrgen, tile_id, 0);
                    local_noc_addrs[valid_tiles] = output_addrgen.get_noc_addr(tile_id);
                    valid_tiles++;
                }
                band_tile_iter++;
            }

            const bool send_remote =
                (direction == 1 && num_targets_backward_direction) || (direction == 0 && num_targets_forward_direction);
            if (valid_tiles > 1) {
                if (send_remote) {
                    fabric_unicast_noc_scatter_write_with_state<
                        UnicastScatterWriteUpdateMask::DstAddrs | UnicastScatterWriteUpdateMask::ChunkSizes |
                        UnicastScatterWriteUpdateMask::PayloadSize>(
                        &mux_connection,
                        pkt_scatter_hdr,
                        l1_read_addr,
                        NocUnicastScatterCommandHeader(noc_addrs, chunk_sizes, valid_tiles),
                        output_page_size * valid_tiles);
                }
                if (direction == 1 && write_local) {
                    size_t local_l1_read_addr = l1_read_addr;
                    for (uint32_t k = 0; k < valid_tiles; ++k) {
                        noc_async_write(local_l1_read_addr, local_noc_addrs[k], output_page_size);
                        local_l1_read_addr += output_page_size;
                    }
                    noc_obj.async_write_barrier();
                }
            } else if (valid_tiles == 1) {
                if (send_remote) {
                    fabric_unicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                        &mux_connection, pkt_unicast_hdr, l1_read_addr, NocUnicastCommandHeader{noc_addrs[0]});
                }
                if (direction == 1 && write_local) {
                    noc_async_write(l1_read_addr, local_noc_addrs[0], output_page_size);
                    noc_obj.async_write_barrier();
                }
            }
            noc_obj.async_writes_flushed();
            cb_output.pop_front(max_tiles_per_packet);
        }
        // One matmul-aggregator inc per band, ordered after this band's writes on the fabric.
        if (mm_signal_enabled) {
            fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                &mux_connection,
                pkt_hdr_mm_sem_inc,
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{mm_agg_sem_noc_addr, 0, 0});
        }
    }
    // Write the semaphore packet
    if ((direction == 1 && num_targets_backward_direction) || (direction == 0 && num_targets_forward_direction)) {
        fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            &mux_connection,
            pkt_hdr_sem_inc,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt, 0, 0});
    }

    uint32_t new_chunk_start_tile = chunk_start_tile + chunk_width;
    uint32_t new_chunk_row = new_chunk_start_tile / input_tensor_Wt;
    if (new_chunk_row != chunk_start_row) {
        chunk_start_tile = (chunk_start_row + subchunk_height) * input_tensor_Wt;
    } else {
        chunk_start_tile = new_chunk_start_tile;
    }
    return 0;  // return value is unused by callers
}
