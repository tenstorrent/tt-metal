// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "../primitives/interleave.hpp"
#include "../primitives/noc_local.hpp"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/wavelet/common/storage_contract.hpp"
#include "ttnn/operations/wavelet/device/protocol/lwt_config.hpp"
#include "ttnn/operations/wavelet/planner/step.hpp"

namespace {

using ttnn::operations::wavelet::kernels::primitives::write_direct_interleaved_signal;
using ttnn::operations::wavelet::kernels::primitives::write_reconstructed_signal;

ALWI uint32_t resolve_workspace_slot(
    const uint32_t slot,
    const uint32_t workspace_a_addr,
    const uint32_t workspace_b_addr,
    const uint32_t workspace_scratch_addr) {
    return slot == 0 ? workspace_a_addr : (slot == 1 ? workspace_b_addr : workspace_scratch_addr);
}

template <typename ConfigAccessor>
ALWI const uint32_t* load_route_config(
    const ConfigAccessor& config, const uint32_t config_addr, const uint32_t cb_config, const uint32_t page_index) {
    const auto page_accessor =
        TensorAccessor(config, config_addr, ttnn::operations::wavelet::device_protocol::kRouteConfigPageBytes);
    CircularBuffer config_buffer(cb_config);
    Noc noc;

    config_buffer.reserve_back(1);
    noc.async_read(
        page_accessor,
        config_buffer,
        ttnn::operations::wavelet::device_protocol::kRouteConfigPageBytes,
        {.page_id = page_index},
        {});
    noc.async_read_barrier();
    config_buffer.push_back(1);
    config_buffer.wait_front(1);
    return reinterpret_cast<const uint32_t*>(config_buffer.get_read_ptr());
}

template <typename DstAccessor>
ALWI void write_dram_half_block(
    const DstAccessor& dst,
    const uint32_t tile_addr,
    const uint32_t row,
    const uint32_t output_page,
    const uint32_t local_output_index,
    const uint32_t output_offset,
    const uint32_t output_length) {
    if (local_output_index >= output_length) {
        return;
    }
    constexpr uint32_t block_elements = ttnn::operations::wavelet::device_protocol::kLwtHalfStickElements;
    const uint32_t remaining = output_length - local_output_index;
    const uint32_t logical_block_elements = remaining < block_elements ? remaining : block_elements;
    const uint32_t destination_index = output_offset + local_output_index;
    const uint32_t destination_stick = destination_index / ttnn::operations::wavelet::kStickWidth;
    const uint32_t destination_lane = destination_index % ttnn::operations::wavelet::kStickWidth;
    const uint32_t source_offset = row * ttnn::operations::wavelet::device_protocol::kLwtHalfStickBytes;
    Noc noc;
    noc.async_write(
        CoreLocalMem<uint32_t>(tile_addr + source_offset),
        dst,
        logical_block_elements * sizeof(float),
        {},
        {.page_id = output_page + destination_stick, .offset_bytes = destination_lane * sizeof(float)});
}

template <typename DstAccessor>
ALWI void write_dram_output_groups(
    const DstAccessor& dst,
    const uint32_t cb_output,
    const uint32_t tile_bytes,
    const uint32_t output_page,
    const uint32_t output_offset,
    const uint32_t output_length,
    const uint32_t group_count) {
    CircularBuffer output_buffer(cb_output);
    Noc noc;
    for (uint32_t group = 0; group < group_count; ++group) {
        output_buffer.wait_front(3);
        const uint32_t output_tiles = output_buffer.get_read_ptr();
        const uint32_t group_base = group * ttnn::operations::wavelet::device_protocol::kLwtGroupOutputElements;

        for (uint32_t row = 0; row < ttnn::operations::wavelet::device_protocol::kLwtRowsPerGroup; ++row) {
            const uint32_t row_base =
                group_base + row * ttnn::operations::wavelet::device_protocol::kLwtOutputBlocksPerRow *
                                 ttnn::operations::wavelet::device_protocol::kLwtHalfStickElements;
            for (uint32_t block = 0; block < ttnn::operations::wavelet::device_protocol::kLwtOutputBlocksPerRow;
                 ++block) {
                write_dram_half_block(
                    dst,
                    output_tiles + block * tile_bytes,
                    row,
                    output_page,
                    row_base + block * ttnn::operations::wavelet::device_protocol::kLwtHalfStickElements,
                    output_offset,
                    output_length);
            }
        }
        noc.async_write_barrier();
        output_buffer.pop_front(3);
    }
}

template <bool UseNocLocalWrite>
ALWI void write_local_half_block(
    const Noc& noc,
    const UnicastEndpoint& local_endpoint,
    const ttnn::operations::wavelet::kernels::primitives::LocalNocCoordinates& local_coordinates,
    const uint32_t dst_addr,
    const uint32_t tile_addr,
    const uint32_t row,
    const uint32_t local_output_index,
    const uint32_t output_offset,
    const uint32_t output_length) {
    if (local_output_index >= output_length) {
        return;
    }

    const uint32_t source_index = row * ttnn::operations::wavelet::device_protocol::kLwtHalfStickElements;
    const uint32_t destination_index = output_offset + local_output_index;
    if constexpr (UseNocLocalWrite) {
        noc.async_write_with_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
            CoreLocalMem<uint32_t>(tile_addr + source_index * static_cast<uint32_t>(sizeof(float))),
            local_endpoint,
            ttnn::operations::wavelet::device_protocol::kLwtHalfStickBytes,
            {},
            ttnn::operations::wavelet::kernels::primitives::local_noc_destination(
                local_coordinates, dst_addr + destination_index * static_cast<uint32_t>(sizeof(float))));
    } else {
        auto* dst = reinterpret_cast<volatile tt_l1_ptr float*>(dst_addr);
        const auto* src = reinterpret_cast<volatile tt_l1_ptr float*>(tile_addr);
#pragma GCC unroll 8
        for (uint32_t lane = 0; lane < ttnn::operations::wavelet::device_protocol::kLwtHalfStickElements; ++lane) {
            dst[destination_index + lane] = src[source_index + lane];
        }
    }
}

template <bool UseNocLocalWrite, bool TileNative, bool HybridTileMirror>
ALWI void write_local_output_groups(
    const uint32_t dst_addr,
    const uint32_t cb_output,
    const uint32_t tile_bytes,
    const uint32_t output_offset,
    const uint32_t output_length,
    const uint32_t group_count,
    const uint32_t tile_mirror_offset,
    const bool write_tile_mirror) {
    constexpr uint32_t group_elements = ttnn::operations::wavelet::device_protocol::kLwtGroupOutputElements;
    constexpr uint32_t blocks_per_group = ttnn::operations::wavelet::device_protocol::kLwtOutputBlocksPerRow;
    const uint32_t group_bytes = blocks_per_group * tile_bytes;
    CircularBuffer output_buffer(cb_output);
    Noc noc;
    UnicastEndpoint local_endpoint;
    const auto local_coordinates = ttnn::operations::wavelet::kernels::primitives::local_noc_coordinates(noc);
    if constexpr (UseNocLocalWrite && !HybridTileMirror) {
        const uint32_t write_bytes =
            TileNative ? tile_bytes : ttnn::operations::wavelet::device_protocol::kLwtHalfStickBytes;
        noc.set_async_write_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
            local_endpoint,
            write_bytes,
            ttnn::operations::wavelet::kernels::primitives::local_noc_destination(local_coordinates, dst_addr));
    }

    for (uint32_t group = 0; group < group_count; ++group) {
        output_buffer.wait_front(3);
        const uint32_t output_tiles = output_buffer.get_read_ptr();
        const uint32_t group_base = group * ttnn::operations::wavelet::device_protocol::kLwtGroupOutputElements;
        if constexpr (UseNocLocalWrite && HybridTileMirror) {
            const uint32_t write_bytes =
                TileNative ? tile_bytes : ttnn::operations::wavelet::device_protocol::kLwtHalfStickBytes;
            noc.set_async_write_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                local_endpoint,
                write_bytes,
                ttnn::operations::wavelet::kernels::primitives::local_noc_destination(local_coordinates, dst_addr));
        }
        if constexpr (TileNative) {
            const uint32_t destination_index = output_offset + group_base;
            const uint32_t destination_group = destination_index / group_elements;
            const uint32_t destination_addr = dst_addr + destination_group * group_bytes;
            if constexpr (UseNocLocalWrite) {
#pragma GCC unroll 3
                for (uint32_t block = 0; block < blocks_per_group; ++block) {
                    noc.async_write_with_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                        CoreLocalMem<uint32_t>(output_tiles + block * tile_bytes),
                        local_endpoint,
                        tile_bytes,
                        {},
                        ttnn::operations::wavelet::kernels::primitives::local_noc_destination(
                            local_coordinates, destination_addr + block * tile_bytes));
                }
            } else {
                auto* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(destination_addr);
                const auto* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(output_tiles);
#pragma GCC unroll 4
                for (uint32_t word = 0; word < group_bytes / sizeof(uint32_t); ++word) {
                    dst[word] = src[word];
                }
            }
        } else {
            for (uint32_t row = 0; row < ttnn::operations::wavelet::device_protocol::kLwtRowsPerGroup; ++row) {
                const uint32_t row_base =
                    group_base +
                    row * blocks_per_group * ttnn::operations::wavelet::device_protocol::kLwtHalfStickElements;
                for (uint32_t block = 0; block < blocks_per_group; ++block) {
                    write_local_half_block<UseNocLocalWrite>(
                        noc,
                        local_endpoint,
                        local_coordinates,
                        dst_addr,
                        output_tiles + block * tile_bytes,
                        row,
                        row_base + block * ttnn::operations::wavelet::device_protocol::kLwtHalfStickElements,
                        output_offset,
                        output_length);
                }
            }
        }
        if constexpr (HybridTileMirror && !TileNative) {
            const uint32_t destination_index = output_offset + group_base;
            if (write_tile_mirror && destination_index % group_elements == 0) {
                const uint32_t destination_group = destination_index / group_elements;
                const uint32_t destination_addr = dst_addr + tile_mirror_offset + destination_group * group_bytes;
                if constexpr (UseNocLocalWrite) {
                    noc.set_async_write_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                        local_endpoint,
                        tile_bytes,
                        ttnn::operations::wavelet::kernels::primitives::local_noc_destination(
                            local_coordinates, destination_addr));
#pragma GCC unroll 3
                    for (uint32_t block = 0; block < blocks_per_group; ++block) {
                        noc.async_write_with_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                            CoreLocalMem<uint32_t>(output_tiles + block * tile_bytes),
                            local_endpoint,
                            tile_bytes,
                            {},
                            ttnn::operations::wavelet::kernels::primitives::local_noc_destination(
                                local_coordinates, destination_addr + block * tile_bytes));
                    }
                } else {
                    auto* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(destination_addr);
                    const auto* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(output_tiles);
#pragma GCC unroll 4
                    for (uint32_t word = 0; word < group_bytes / sizeof(uint32_t); ++word) {
                        dst[word] = src[word];
                    }
                }
            }
        }
        if constexpr (UseNocLocalWrite) {
            noc.async_writes_flushed();
        }
        output_buffer.pop_front(3);
    }
}

}  // namespace

void kernel_main() {
    const uint32_t route_config_addr = get_arg_val<uint32_t>(0);
    const uint32_t chunk_begin = get_arg_val<uint32_t>(1);
    const uint32_t chunk_count = get_arg_val<uint32_t>(2);
    const uint32_t route_count = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_config = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output = get_compile_time_arg_val(1);
    constexpr uint32_t cb_sync = get_compile_time_arg_val(2);
    constexpr bool use_noc_local_write = get_compile_time_arg_val(3) != 0;
    constexpr bool tile_native_workspace = get_compile_time_arg_val(4) != 0;
    constexpr bool inverse = get_compile_time_arg_val(5) != 0;
    constexpr uint32_t cb_interleave = get_compile_time_arg_val(6);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(7);
    constexpr uint32_t interleave_batch_sticks = get_compile_time_arg_val(8);
    constexpr bool hybrid_tile_mirror = get_compile_time_arg_val(9) != 0;
    constexpr uint32_t cb_workspace_a = get_compile_time_arg_val(10);
    constexpr uint32_t cb_workspace_b = get_compile_time_arg_val(11);
    constexpr uint32_t cb_workspace_scratch = get_compile_time_arg_val(12);
    constexpr uint32_t tile_bytes = get_tile_size(cb_output);
    constexpr auto config_args = TensorAccessorArgs<13>();
    constexpr auto final_args = TensorAccessorArgs<config_args.next_compile_time_args_offset()>();
    CircularBuffer config_buffer(cb_config);
    CircularBuffer sync_buffer(cb_sync);
    const uint32_t workspace_a_addr = CircularBuffer(cb_workspace_a).get_write_ptr();
    const uint32_t workspace_b_addr = CircularBuffer(cb_workspace_b).get_write_ptr();
    const uint32_t workspace_scratch_addr = CircularBuffer(cb_workspace_scratch).get_write_ptr();
    Noc noc;

    if constexpr (inverse) {
        const uint32_t chunk_config_addr = get_arg_val<uint32_t>(4);
        const uint32_t output_addr = get_arg_val<uint32_t>(5);
        const uint32_t left_pad = get_arg_val<uint32_t>(6);
        const uint32_t tile_mirror_offset = get_arg_val<uint32_t>(7);
        const uint32_t chunks_per_sample = get_arg_val<uint32_t>(8);
        const uint32_t output_pages_per_sample = get_arg_val<uint32_t>(9);
        const auto output = TensorAccessor(final_args, output_addr, output_page_size);
        for (uint32_t local_chunk = 0; local_chunk < chunk_count; ++local_chunk) {
            const uint32_t global_work_item = chunk_begin + local_chunk;
            const uint32_t batch_index = global_work_item / chunks_per_sample;
            const uint32_t global_chunk = global_work_item - batch_index * chunks_per_sample;
            const uint32_t output_page = batch_index * output_pages_per_sample;
            uint32_t chunk_words[ttnn::operations::wavelet::device_protocol::kLwtChunkConfigWordCount];
            const uint32_t* loaded_chunk = load_route_config(config_args, chunk_config_addr, cb_config, global_chunk);
#pragma GCC unroll 8
            for (uint32_t word = 0; word < ttnn::operations::wavelet::device_protocol::kLwtChunkConfigWordCount;
                 ++word) {
                chunk_words[word] = loaded_chunk[word];
            }
            config_buffer.pop_front(1);
            chunk_words[ttnn::operations::wavelet::device_protocol::kIlwtFinalEvenAddr] = resolve_workspace_slot(
                chunk_words[ttnn::operations::wavelet::device_protocol::kIlwtFinalEvenAddr],
                workspace_a_addr,
                workspace_b_addr,
                workspace_scratch_addr);
            chunk_words[ttnn::operations::wavelet::device_protocol::kIlwtFinalOddAddr] = resolve_workspace_slot(
                chunk_words[ttnn::operations::wavelet::device_protocol::kIlwtFinalOddAddr],
                workspace_a_addr,
                workspace_b_addr,
                workspace_scratch_addr);

            bool direct_interleave_written = false;
            for (uint32_t route_index = 0; route_index < route_count; ++route_index) {
                const uint32_t config_index = global_chunk * route_count + route_index;
                const uint32_t* route = load_route_config(config_args, route_config_addr, cb_config, config_index);
                const uint32_t route_flags = route[ttnn::operations::wavelet::device_protocol::kRouteFlags];
                const bool direct_interleave =
                    (route_flags & ttnn::operations::wavelet::device_protocol::kRouteFlagIlwtFinalInterleave) != 0;
                if (direct_interleave) {
                    write_direct_interleaved_signal<tile_native_workspace, interleave_batch_sticks>(
                        output,
                        output_page,
                        cb_output,
                        cb_interleave,
                        tile_bytes,
                        left_pad,
                        route[ttnn::operations::wavelet::device_protocol::kRouteType],
                        route[ttnn::operations::wavelet::device_protocol::kRouteGroupCount],
                        chunk_words[ttnn::operations::wavelet::device_protocol::kIlwtFinalEvenAddr],
                        chunk_words[ttnn::operations::wavelet::device_protocol::kIlwtFinalEvenOffset],
                        chunk_words[ttnn::operations::wavelet::device_protocol::kIlwtFinalEvenBegin],
                        chunk_words[ttnn::operations::wavelet::device_protocol::kIlwtFinalOddAddr],
                        chunk_words[ttnn::operations::wavelet::device_protocol::kIlwtFinalOddOffset],
                        chunk_words[ttnn::operations::wavelet::device_protocol::kIlwtFinalOddBegin],
                        chunk_words[ttnn::operations::wavelet::device_protocol::kIlwtOutputBegin],
                        chunk_words[ttnn::operations::wavelet::device_protocol::kIlwtOutputLength]);
                    direct_interleave_written = true;
                } else {
                    write_local_output_groups<use_noc_local_write, tile_native_workspace, hybrid_tile_mirror>(
                        resolve_workspace_slot(
                            route[ttnn::operations::wavelet::device_protocol::kRouteOutputAddr],
                            workspace_a_addr,
                            workspace_b_addr,
                            workspace_scratch_addr),
                        cb_output,
                        tile_bytes,
                        route[ttnn::operations::wavelet::device_protocol::kRouteOutputOffset],
                        route[ttnn::operations::wavelet::device_protocol::kRouteOutputLength],
                        route[ttnn::operations::wavelet::device_protocol::kRouteGroupCount],
                        tile_mirror_offset,
                        (route_flags & ttnn::operations::wavelet::device_protocol::kRouteFlagOutputTileMirror) != 0);
                }
                noc.async_write_barrier();
                config_buffer.pop_front(1);
                if (route_index + 1 < route_count) {
                    sync_buffer.reserve_back(1);
                    sync_buffer.push_back(1);
                }
            }

            if (!direct_interleave_written) {
                write_reconstructed_signal<tile_native_workspace, interleave_batch_sticks>(
                    output,
                    output_page,
                    cb_interleave,
                    left_pad,
                    chunk_words[ttnn::operations::wavelet::device_protocol::kIlwtFinalEvenAddr],
                    chunk_words[ttnn::operations::wavelet::device_protocol::kIlwtFinalEvenOffset],
                    chunk_words[ttnn::operations::wavelet::device_protocol::kIlwtFinalEvenBegin],
                    chunk_words[ttnn::operations::wavelet::device_protocol::kIlwtFinalOddAddr],
                    chunk_words[ttnn::operations::wavelet::device_protocol::kIlwtFinalOddOffset],
                    chunk_words[ttnn::operations::wavelet::device_protocol::kIlwtFinalOddBegin],
                    chunk_words[ttnn::operations::wavelet::device_protocol::kIlwtOutputBegin],
                    chunk_words[ttnn::operations::wavelet::device_protocol::kIlwtOutputLength]);
            }
            if (local_chunk + 1 < chunk_count) {
                sync_buffer.reserve_back(1);
                sync_buffer.push_back(1);
            }
        }
    } else {
        const uint32_t final_even_addr = get_arg_val<uint32_t>(4);
        const uint32_t final_odd_addr = get_arg_val<uint32_t>(5);
        const uint32_t tile_mirror_offset = get_arg_val<uint32_t>(6);
        const uint32_t chunks_per_sample = get_arg_val<uint32_t>(7);
        const uint32_t output_pages_per_sample = get_arg_val<uint32_t>(8);
        const uint32_t local_route_count = chunk_count * route_count;
        uint32_t flattened_route = 0;
        for (uint32_t local_chunk = 0; local_chunk < chunk_count; ++local_chunk) {
            const uint32_t global_work_item = chunk_begin + local_chunk;
            const uint32_t batch_index = global_work_item / chunks_per_sample;
            const uint32_t global_chunk = global_work_item - batch_index * chunks_per_sample;
            const uint32_t output_page = batch_index * output_pages_per_sample;
            for (uint32_t route_index = 0; route_index < route_count; ++route_index, ++flattened_route) {
                const uint32_t config_index = global_chunk * route_count + route_index;
                const uint32_t* route = load_route_config(config_args, route_config_addr, cb_config, config_index);
                uint32_t output_addr = route[ttnn::operations::wavelet::device_protocol::kRouteOutputAddr];
                const uint32_t output_length = route[ttnn::operations::wavelet::device_protocol::kRouteOutputLength];
                const uint32_t output_offset = route[ttnn::operations::wavelet::device_protocol::kRouteOutputOffset];
                const uint32_t group_count = route[ttnn::operations::wavelet::device_protocol::kRouteGroupCount];
                const uint32_t route_flags = route[ttnn::operations::wavelet::device_protocol::kRouteFlags];
                const bool final_dram =
                    (route_flags & ttnn::operations::wavelet::device_protocol::kRouteFlagFinalDram) != 0;
                if (final_dram) {
                    output_addr = (route_flags & ttnn::operations::wavelet::device_protocol::kRouteFlagFinalEven) != 0
                                      ? final_even_addr
                                      : final_odd_addr;
                    const auto dst = TensorAccessor(final_args, output_addr, output_page_size);
                    write_dram_output_groups(
                        dst, cb_output, tile_bytes, output_page, output_offset, output_length, group_count);
                } else {
                    write_local_output_groups<use_noc_local_write, tile_native_workspace, hybrid_tile_mirror>(
                        resolve_workspace_slot(output_addr, workspace_a_addr, workspace_b_addr, workspace_scratch_addr),
                        cb_output,
                        tile_bytes,
                        output_offset,
                        output_length,
                        group_count,
                        tile_mirror_offset,
                        (route_flags & ttnn::operations::wavelet::device_protocol::kRouteFlagOutputTileMirror) != 0);
                }

                noc.async_write_barrier();
                config_buffer.pop_front(1);
                if (flattened_route + 1 < local_route_count) {
                    sync_buffer.reserve_back(1);
                    sync_buffer.push_back(1);
                }
            }
        }
    }
}
