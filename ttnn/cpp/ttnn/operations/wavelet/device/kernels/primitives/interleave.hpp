// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/wavelet/common/storage_contract.hpp"
#include "ttnn/operations/wavelet/device/protocol/lwt_config.hpp"
#include "ttnn/operations/wavelet/planner/step.hpp"
#include "workspace_layout.hpp"

namespace ttnn::operations::wavelet::kernels::primitives {

template <bool TileNative, uint32_t BatchSticks, typename DstAccessor>
ALWI void write_reconstructed_signal(
    const DstAccessor& dst,
    const uint32_t output_page,
    const uint32_t cb_interleave,
    const uint32_t left_pad,
    const uint32_t even_addr,
    const uint32_t even_offset,
    const uint32_t even_begin,
    const uint32_t odd_addr,
    const uint32_t odd_offset,
    const uint32_t odd_begin,
    const uint32_t output_begin,
    const uint32_t output_length) {
    CircularBuffer interleave_buffer(cb_interleave);
    Noc noc;
    const auto* even = reinterpret_cast<volatile tt_l1_ptr float*>(even_addr);
    const auto* odd = reinterpret_cast<volatile tt_l1_ptr float*>(odd_addr);
    const uint32_t output_end = output_begin + output_length;
    const uint32_t first_stick = output_begin / ttnn::operations::wavelet::kStickWidth;
    const uint32_t stick_count =
        (output_length + ttnn::operations::wavelet::kStickWidth - 1U) / ttnn::operations::wavelet::kStickWidth;

    static_assert(BatchSticks > 0, "ILWT interleave batch must be non-zero");
    for (uint32_t batch_begin = 0; batch_begin < stick_count; batch_begin += BatchSticks) {
        const uint32_t batch_count = stick_count - batch_begin < BatchSticks ? stick_count - batch_begin : BatchSticks;
        interleave_buffer.reserve_back(batch_count);
        const uint32_t staging_base = interleave_buffer.get_write_ptr();
        for (uint32_t batch_stick = 0; batch_stick < batch_count; ++batch_stick) {
            const uint32_t local_stick = batch_begin + batch_stick;
            auto* staging = reinterpret_cast<float*>(
                staging_base + batch_stick * ttnn::operations::wavelet::device_protocol::kStickBytes);
            const uint32_t signal_base = (first_stick + local_stick) * ttnn::operations::wavelet::kStickWidth;
#pragma GCC unroll 8
            for (uint32_t lane = 0; lane < ttnn::operations::wavelet::kStickWidth; ++lane) {
                const uint32_t signal_index = signal_base + lane;
                float value = 0.0F;
                if (signal_index >= output_begin && signal_index < output_end) {
                    const uint32_t padded_index = left_pad + signal_index;
                    const uint32_t split_index = padded_index / 2U;
                    if ((padded_index & 1U) == 0) {
                        const uint32_t logical_index = even_offset + split_index - even_begin;
                        value = even[workspace_physical_index<TileNative>(logical_index)];
                    } else {
                        const uint32_t logical_index = odd_offset + split_index - odd_begin;
                        value = odd[workspace_physical_index<TileNative>(logical_index)];
                    }
                }
                staging[lane] = value;
            }
            noc.async_write(
                CoreLocalMem<uint32_t>(
                    staging_base + batch_stick * ttnn::operations::wavelet::device_protocol::kStickBytes),
                dst,
                ttnn::operations::wavelet::device_protocol::kStickBytes,
                {},
                {.page_id = output_page + first_stick + local_stick});
        }
        noc.async_write_barrier();
        interleave_buffer.push_back(batch_count);
        interleave_buffer.wait_front(batch_count);
        interleave_buffer.pop_front(batch_count);
    }
}

[[nodiscard]] ALWI float read_direct_output_value(
    const uint32_t output_tiles, const uint32_t tile_bytes, const uint32_t logical_index) {
    constexpr uint32_t row_elements = ttnn::operations::wavelet::device_protocol::kLwtOutputBlocksPerRow *
                                      ttnn::operations::wavelet::device_protocol::kLwtHalfStickElements;
    const uint32_t row = logical_index / row_elements;
    const uint32_t in_row = logical_index - row * row_elements;
    const uint32_t block = in_row / ttnn::operations::wavelet::device_protocol::kLwtHalfStickElements;
    const uint32_t lane = in_row - block * ttnn::operations::wavelet::device_protocol::kLwtHalfStickElements;
    const auto* tile = reinterpret_cast<volatile tt_l1_ptr float*>(output_tiles + block * tile_bytes);
    return tile[row * ttnn::operations::wavelet::device_protocol::kLwtHalfStickElements + lane];
}

template <bool TileNative, uint32_t BatchSticks, typename DstAccessor>
ALWI void write_direct_interleaved_signal(
    const DstAccessor& dst,
    const uint32_t output_page,
    const uint32_t cb_output,
    const uint32_t cb_interleave,
    const uint32_t tile_bytes,
    const uint32_t left_pad,
    const uint32_t route_type,
    const uint32_t updated_group_count,
    const uint32_t even_addr,
    const uint32_t even_offset,
    const uint32_t even_begin,
    const uint32_t odd_addr,
    const uint32_t odd_offset,
    const uint32_t odd_begin,
    const uint32_t output_begin,
    const uint32_t output_length) {
    CircularBuffer output_buffer(cb_output);
    CircularBuffer interleave_buffer(cb_interleave);
    Noc noc;
    constexpr uint32_t split_group_elements = ttnn::operations::wavelet::device_protocol::kLwtGroupOutputElements;
    constexpr uint32_t signal_group_elements = ttnn::operations::wavelet::device_protocol::kIlwtGroupOutputElements;
    static_assert(
        signal_group_elements % ttnn::operations::wavelet::kStickWidth == 0,
        "ILWT direct-interleave groups must be stick aligned");
    const bool updates_even = route_type == static_cast<uint32_t>(ttnn::operations::wavelet::StepType::kUpdate);
    const auto* even = reinterpret_cast<volatile tt_l1_ptr float*>(even_addr);
    const auto* odd = reinterpret_cast<volatile tt_l1_ptr float*>(odd_addr);
    const uint32_t output_group_count = (output_length + signal_group_elements - 1U) / signal_group_elements;

    for (uint32_t group = 0; group < output_group_count; ++group) {
        const bool has_updated_values = group < updated_group_count;
        uint32_t output_tiles = 0;
        if (has_updated_values) {
            output_buffer.wait_front(3);
            output_tiles = output_buffer.get_read_ptr();
        }

        const uint32_t group_signal_offset = group * signal_group_elements;
        const uint32_t group_output_length = output_length - group_signal_offset < signal_group_elements
                                                 ? output_length - group_signal_offset
                                                 : signal_group_elements;
        const uint32_t group_begin = output_begin + group_signal_offset;
        const uint32_t group_end = group_begin + group_output_length;
        const uint32_t first_stick = group_begin / ttnn::operations::wavelet::kStickWidth;
        const uint32_t last_stick =
            (group_end + ttnn::operations::wavelet::kStickWidth - 1U) / ttnn::operations::wavelet::kStickWidth;
        const uint32_t stick_count = last_stick - first_stick;

        static_assert(BatchSticks > 0, "ILWT direct-interleave batch must be non-zero");
        for (uint32_t batch_begin = 0; batch_begin < stick_count; batch_begin += BatchSticks) {
            const uint32_t batch_count =
                stick_count - batch_begin < BatchSticks ? stick_count - batch_begin : BatchSticks;
            interleave_buffer.reserve_back(batch_count);
            const uint32_t staging_base = interleave_buffer.get_write_ptr();
            for (uint32_t batch_stick = 0; batch_stick < batch_count; ++batch_stick) {
                const uint32_t local_stick = batch_begin + batch_stick;
                auto* staging = reinterpret_cast<float*>(
                    staging_base + batch_stick * ttnn::operations::wavelet::device_protocol::kStickBytes);
                const uint32_t signal_base = (first_stick + local_stick) * ttnn::operations::wavelet::kStickWidth;
#pragma GCC unroll 8
                for (uint32_t lane = 0; lane < ttnn::operations::wavelet::kStickWidth; ++lane) {
                    const uint32_t signal_index = signal_base + lane;
                    float value = 0.0F;
                    if (signal_index >= group_begin && signal_index < group_end) {
                        const uint32_t local_signal_index = signal_index - output_begin;
                        const uint32_t padded_index = left_pad + signal_index;
                        const uint32_t split_index = padded_index / 2U;
                        const bool is_even = (padded_index & 1U) == 0;
                        if (is_even == updates_even) {
                            const uint32_t updated_begin = updates_even ? even_begin : odd_begin;
                            const uint32_t local_updated_index = split_index - updated_begin;
                            const int32_t group_updated_index = static_cast<int32_t>(local_updated_index) -
                                                                static_cast<int32_t>(group * split_group_elements);
                            if (has_updated_values && group_updated_index >= 0 &&
                                group_updated_index < static_cast<int32_t>(split_group_elements)) {
                                value = read_direct_output_value(
                                    output_tiles, tile_bytes, static_cast<uint32_t>(group_updated_index));
                            } else {
                                const uint32_t logical_index = (updates_even ? even_offset : odd_offset) + split_index -
                                                               (updates_even ? even_begin : odd_begin);
                                value =
                                    (updates_even ? even : odd)[workspace_physical_index<TileNative>(logical_index)];
                            }
                        } else if (is_even) {
                            const uint32_t logical_index = even_offset + split_index - even_begin;
                            value = even[workspace_physical_index<TileNative>(logical_index)];
                        } else {
                            const uint32_t logical_index = odd_offset + split_index - odd_begin;
                            value = odd[workspace_physical_index<TileNative>(logical_index)];
                        }
                    }
                    staging[lane] = value;
                }
                noc.async_write(
                    CoreLocalMem<uint32_t>(
                        staging_base + batch_stick * ttnn::operations::wavelet::device_protocol::kStickBytes),
                    dst,
                    ttnn::operations::wavelet::device_protocol::kStickBytes,
                    {},
                    {.page_id = output_page + first_stick + local_stick});
            }
            noc.async_write_barrier();
            interleave_buffer.push_back(batch_count);
            interleave_buffer.wait_front(batch_count);
            interleave_buffer.pop_front(batch_count);
        }

        if (has_updated_values) {
            output_buffer.pop_front(3);
        }
    }
}

}  // namespace ttnn::operations::wavelet::kernels::primitives
