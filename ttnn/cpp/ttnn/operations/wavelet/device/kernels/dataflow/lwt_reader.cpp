// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "../primitives/noc_local.hpp"
#include "../primitives/stick_cache.hpp"
#include "../primitives/workspace_layout.hpp"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/wavelet/common/boundary.hpp"
#include "ttnn/operations/wavelet/common/storage_contract.hpp"
#include "ttnn/operations/wavelet/device/protocol/lwt_config.hpp"
#include "ttnn/operations/wavelet/planner/step.hpp"

namespace {

constexpr uint32_t kStepPredict = static_cast<uint32_t>(ttnn::operations::wavelet::StepType::kPredict);
constexpr uint32_t kStepUpdate = static_cast<uint32_t>(ttnn::operations::wavelet::StepType::kUpdate);
constexpr uint32_t kStepScaleEven = static_cast<uint32_t>(ttnn::operations::wavelet::StepType::kScaleEven);
constexpr uint32_t kStepScaleOdd = static_cast<uint32_t>(ttnn::operations::wavelet::StepType::kScaleOdd);

constexpr uint32_t kBlockElements = ttnn::operations::wavelet::device_protocol::kLwtHalfStickElements;
constexpr uint32_t kRowsPerGroup = ttnn::operations::wavelet::device_protocol::kLwtRowsPerGroup;
constexpr uint32_t kOutputBlocksPerRow = ttnn::operations::wavelet::device_protocol::kLwtOutputBlocksPerRow;
constexpr uint32_t kGroupOutputElements = ttnn::operations::wavelet::device_protocol::kLwtGroupOutputElements;
constexpr uint32_t kSourcePackedElements = kGroupOutputElements + kBlockElements;
constexpr uint32_t kNarrowTileElements = ttnn::operations::wavelet::device_protocol::kLwtNarrowTileElements;
constexpr uint32_t kNarrowTileBytes = ttnn::operations::wavelet::device_protocol::kLwtNarrowTileBytes;
constexpr uint32_t kGroupOutputBytes = kGroupOutputElements * sizeof(float);
constexpr uint32_t kNocL1ReadAlignmentElements = NOC_L1_READ_ALIGNMENT_BYTES / sizeof(float);
static_assert(NOC_L1_READ_ALIGNMENT_BYTES % sizeof(float) == 0);

using ttnn::operations::wavelet::device_protocol::config_word_index;
using ttnn::operations::wavelet::device_protocol::LwtChunkConfigWord;
using ttnn::operations::wavelet::device_protocol::RouteConfigWord;
using ttnn::operations::wavelet::kernels::primitives::WorkspaceIndexCursor;

ALWI uint32_t resolve_workspace_slot(
    const uint32_t slot,
    const uint32_t workspace_a_addr,
    const uint32_t workspace_b_addr,
    const uint32_t workspace_scratch_addr) {
    return slot == 0 ? workspace_a_addr : (slot == 1 ? workspace_b_addr : workspace_scratch_addr);
}

ALWI void read_workspace_block(const volatile tt_l1_ptr float* src, WorkspaceIndexCursor& cursor, float* dst) {
    const uint32_t initial_lane = cursor.lane;
    const uint32_t first_count = kBlockElements - initial_lane;
    for (uint32_t index = 0; index < first_count; ++index) {
        dst[index] = src[cursor.physical + index];
    }

    cursor.lane = 0;
    cursor.advance_block();
    for (uint32_t index = 0; index < initial_lane; ++index) {
        dst[first_count + index] = src[cursor.physical + index];
    }
    cursor.lane = initial_lane;
    cursor.physical += initial_lane;
}

template <bool BoundsChecked>
ALWI void read_workspace_block(
    const volatile tt_l1_ptr float* src, const int32_t logical_start, const uint32_t logical_end, float* dst) {
    if constexpr (BoundsChecked) {
        const uint32_t negative_magnitude = 0U - static_cast<uint32_t>(logical_start);
        const uint32_t zero_prefix =
            logical_start < 0 ? (negative_magnitude < kBlockElements ? negative_magnitude : kBlockElements) : 0;
        const uint32_t valid_start = logical_start < 0 ? 0U : static_cast<uint32_t>(logical_start);
        WorkspaceIndexCursor cursor(valid_start);
#pragma GCC unroll 8
        for (uint32_t lane = 0; lane < kBlockElements; ++lane) {
            const bool valid = lane >= zero_prefix && valid_start + lane - zero_prefix < logical_end;
            dst[lane] = valid ? src[cursor.physical] : 0.0F;
            if (valid) {
                cursor.advance();
            }
        }
    } else {
        WorkspaceIndexCursor cursor(static_cast<uint32_t>(logical_start));
#pragma GCC unroll 8
        for (uint32_t lane = 0; lane < kBlockElements; ++lane) {
            dst[lane] = src[cursor.physical];
            cursor.advance();
        }
    }
}

ALWI void read_aligned_source_group(
    const uint32_t source_addr,
    const uint32_t logical_start,
    const uint32_t src_tiles01_addr,
    const uint32_t src_tiles23_addr) {
    const uint32_t physical_group_addr = source_addr + logical_start * sizeof(float);
    Noc noc;
    UnicastEndpoint local_endpoint;
    const auto local_coordinates = ttnn::operations::wavelet::kernels::primitives::local_noc_coordinates(noc);
    noc.async_read(
        local_endpoint,
        CoreLocalMem<uint32_t>(src_tiles01_addr),
        kNarrowTileBytes,
        ttnn::operations::wavelet::kernels::primitives::local_noc_source(local_coordinates, physical_group_addr),
        {});
    noc.async_read(
        local_endpoint,
        CoreLocalMem<uint32_t>(src_tiles01_addr + kNarrowTileBytes),
        kNarrowTileBytes,
        ttnn::operations::wavelet::kernels::primitives::local_noc_source(
            local_coordinates, physical_group_addr + kNarrowTileBytes),
        {});
    noc.async_read(
        local_endpoint,
        CoreLocalMem<uint32_t>(src_tiles23_addr),
        kNarrowTileBytes,
        ttnn::operations::wavelet::kernels::primitives::local_noc_source(
            local_coordinates, physical_group_addr + 2 * kNarrowTileBytes),
        {});

    noc.async_read(
        local_endpoint,
        CoreLocalMem<uint32_t>(src_tiles23_addr + kNarrowTileBytes),
        kNarrowTileBytes - kBlockElements * sizeof(float),
        ttnn::operations::wavelet::kernels::primitives::local_noc_source(
            local_coordinates, physical_group_addr + kBlockElements * sizeof(float)),
        {});
    noc.async_read(
        local_endpoint,
        CoreLocalMem<uint32_t>(src_tiles23_addr + 2 * kNarrowTileBytes - kBlockElements * sizeof(float)),
        kBlockElements * sizeof(float),
        ttnn::operations::wavelet::kernels::primitives::local_noc_source(
            local_coordinates, physical_group_addr + kGroupOutputBytes),
        {});
}

ALWI void read_aligned_output_group(
    const uint32_t source_addr, const uint32_t logical_start, const uint32_t narrow_tiles_addr) {
    const uint32_t physical_group_addr = source_addr + logical_start * sizeof(float);
    Noc noc;
    UnicastEndpoint local_endpoint;
    const auto local_coordinates = ttnn::operations::wavelet::kernels::primitives::local_noc_coordinates(noc);
#pragma GCC unroll 3
    for (uint32_t block = 0; block < kOutputBlocksPerRow; ++block) {
        noc.async_read(
            local_endpoint,
            CoreLocalMem<uint32_t>(narrow_tiles_addr + block * kNarrowTileBytes),
            kNarrowTileBytes,
            ttnn::operations::wavelet::kernels::primitives::local_noc_source(
                local_coordinates, physical_group_addr + block * kNarrowTileBytes),
            {});
    }
}

ALWI void read_row_major_source_group(
    const uint32_t source_addr,
    const uint32_t logical_start,
    const uint32_t src_tiles01_addr,
    const uint32_t src_tiles23_addr) {
    Noc noc;
    UnicastEndpoint local_endpoint;
    const auto local_coordinates = ttnn::operations::wavelet::kernels::primitives::local_noc_coordinates(noc);
    auto local_source =
        ttnn::operations::wavelet::kernels::primitives::local_noc_source(local_coordinates, source_addr);
    noc.set_async_read_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
        local_endpoint, ttnn::operations::wavelet::device_protocol::kLwtHalfStickBytes, local_source);
    for (uint32_t block = 0; block < 4; ++block) {
        const uint32_t destination_tile_addr =
            block < 2 ? src_tiles01_addr + block * kNarrowTileBytes : src_tiles23_addr + (block - 2) * kNarrowTileBytes;
        for (uint32_t row = 0; row < kRowsPerGroup; ++row) {
            const uint32_t source_index = logical_start + (row * kOutputBlocksPerRow + block) * kBlockElements;
            local_source.addr = source_addr + source_index * sizeof(float);
            noc.async_read_with_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                local_endpoint,
                CoreLocalMem<uint32_t>(
                    destination_tile_addr + row * ttnn::operations::wavelet::device_protocol::kLwtHalfStickBytes),
                ttnn::operations::wavelet::device_protocol::kLwtHalfStickBytes,
                local_source,
                {});
        }
    }
}

ALWI void read_row_major_output_group(
    const uint32_t source_addr, const uint32_t logical_start, const uint32_t narrow_tiles_addr) {
    Noc noc;
    UnicastEndpoint local_endpoint;
    const auto local_coordinates = ttnn::operations::wavelet::kernels::primitives::local_noc_coordinates(noc);
    auto local_source =
        ttnn::operations::wavelet::kernels::primitives::local_noc_source(local_coordinates, source_addr);
    noc.set_async_read_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
        local_endpoint, ttnn::operations::wavelet::device_protocol::kLwtHalfStickBytes, local_source);
    for (uint32_t block = 0; block < kOutputBlocksPerRow; ++block) {
        const uint32_t destination_tile_addr = narrow_tiles_addr + block * kNarrowTileBytes;
        for (uint32_t row = 0; row < kRowsPerGroup; ++row) {
            const uint32_t source_index = logical_start + (row * kOutputBlocksPerRow + block) * kBlockElements;
            local_source.addr = source_addr + source_index * sizeof(float);
            noc.async_read_with_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                local_endpoint,
                CoreLocalMem<uint32_t>(
                    destination_tile_addr + row * ttnn::operations::wavelet::device_protocol::kLwtHalfStickBytes),
                ttnn::operations::wavelet::device_protocol::kLwtHalfStickBytes,
                local_source,
                {});
        }
    }
}

template <typename ConfigAccessor>
ALWI const uint32_t* load_config_page(
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

template <ttnn::operations::wavelet::BoundaryMode Boundary, bool TileNative, typename InputAccessor>
ALWI void initialize_lwt_streams(
    const InputAccessor& input,
    const uint32_t even_addr,
    const uint32_t odd_addr,
    const uint32_t cb_input_cache,
    const uint32_t input_length,
    const uint32_t left_pad,
    const uint32_t even_begin,
    const uint32_t even_length,
    const uint32_t odd_begin,
    const uint32_t odd_length,
    const uint32_t input_page,
    const uint32_t input_page_size) {
    ttnn::operations::wavelet::kernels::primitives::StickReadCache input_cache{
        cb_input_cache,
        ttnn::operations::wavelet::device_protocol::kStickBytes,
        ttnn::operations::wavelet::kStickWidth,
        ttnn::operations::wavelet::device_protocol::kLwtCacheStickCount,
        ttnn::operations::wavelet::kernels::primitives::kInvalidStick,
        0,
        0,
        input_page,
        false,
        input_page_size};
    auto* even_dst = reinterpret_cast<volatile tt_l1_ptr float*>(even_addr);
    auto* odd_dst = reinterpret_cast<volatile tt_l1_ptr float*>(odd_addr);
    uint32_t even_written = 0;
    uint32_t odd_written = 0;
    WorkspaceIndexCursor even_cursor(0);
    WorkspaceIndexCursor odd_cursor(0);

    const uint32_t even_end = even_begin + even_length;
    const uint32_t odd_end = odd_begin + odd_length;
    const uint32_t split_begin = even_begin < odd_begin ? even_begin : odd_begin;
    const uint32_t split_end = even_end > odd_end ? even_end : odd_end;
    for (uint32_t split_index = split_begin; split_index < split_end; ++split_index) {
        if (split_index >= even_begin && split_index < even_end) {
            const uint32_t padded_index = 2U * split_index;
            const float value = ttnn::operations::wavelet::kernels::primitives::read_padded_value<Boundary>(
                input, input_cache, input_length, left_pad, padded_index);
            if constexpr (TileNative) {
                even_dst[even_cursor.physical] = value;
                even_cursor.advance();
            } else {
                even_dst[even_written++] = value;
            }
        }
        if (split_index >= odd_begin && split_index < odd_end) {
            const uint32_t padded_index = 2U * split_index + 1U;
            const float value = ttnn::operations::wavelet::kernels::primitives::read_padded_value<Boundary>(
                input, input_cache, input_length, left_pad, padded_index);
            if constexpr (TileNative) {
                odd_dst[odd_cursor.physical] = value;
                odd_cursor.advance();
            } else {
                odd_dst[odd_written++] = value;
            }
        }
    }

    ttnn::operations::wavelet::kernels::primitives::release_cache(input_cache);
}

template <bool TileNative, typename InputAccessor>
ALWI void initialize_inverse_stream(
    const InputAccessor& input,
    const uint32_t input_length,
    const uint32_t input_begin,
    const uint32_t output_addr,
    const uint32_t output_length,
    const uint32_t cb_input_cache,
    const uint32_t input_page,
    const uint32_t input_page_size) {
    ttnn::operations::wavelet::kernels::primitives::StickReadCache input_cache{
        cb_input_cache,
        ttnn::operations::wavelet::device_protocol::kStickBytes,
        ttnn::operations::wavelet::kStickWidth,
        ttnn::operations::wavelet::device_protocol::kLwtCacheStickCount,
        ttnn::operations::wavelet::kernels::primitives::kInvalidStick,
        0,
        0,
        input_page,
        false,
        input_page_size};
    auto* output = reinterpret_cast<volatile tt_l1_ptr float*>(output_addr);
    WorkspaceIndexCursor cursor(0);
    for (uint32_t index = 0; index < output_length; ++index) {
        const float value = ttnn::operations::wavelet::kernels::primitives::read_source_value(
            input, input_cache, input_begin + index, input_length);
        if constexpr (TileNative) {
            output[cursor.physical] = value;
            cursor.advance();
        } else {
            output[index] = value;
        }
    }
    ttnn::operations::wavelet::kernels::primitives::release_cache(input_cache);
}

template <bool BoundsChecked>
ALWI void fill_source_row_major(
    const volatile tt_l1_ptr float* src,
    float* src_tiles01,
    float* src_tiles23,
    const uint32_t source_end,
    const uint32_t source_offset,
    const uint32_t source_left_pad,
    const uint32_t group_base) {
    for (uint32_t block = 0; block < 4; ++block) {
        auto* narrow_tile =
            block < 2 ? src_tiles01 + block * kNarrowTileElements : src_tiles23 + (block - 2) * kNarrowTileElements;
        for (uint32_t row = 0; row < kRowsPerGroup; ++row) {
            auto* tile_row = narrow_tile + row * kBlockElements;
            const int32_t logical_start =
                static_cast<int32_t>(source_offset) - static_cast<int32_t>(source_left_pad) +
                static_cast<int32_t>(group_base + (row * kOutputBlocksPerRow + block) * kBlockElements);
#pragma GCC unroll 8
            for (uint32_t lane = 0; lane < kBlockElements; ++lane) {
                const int32_t logical_index = logical_start + static_cast<int32_t>(lane);
                if constexpr (BoundsChecked) {
                    tile_row[lane] = logical_index >= 0 && static_cast<uint32_t>(logical_index) < source_end
                                         ? src[static_cast<uint32_t>(logical_index)]
                                         : 0.0F;
                } else {
                    tile_row[lane] = src[static_cast<uint32_t>(logical_index)];
                }
            }
        }
    }
}

template <bool BoundsChecked>
ALWI void fill_output_row_major(
    const volatile tt_l1_ptr float* src,
    float* narrow_tiles,
    const uint32_t source_end,
    const uint32_t source_offset,
    const uint32_t output_length,
    const uint32_t group_base) {
    for (uint32_t row = 0; row < kRowsPerGroup; ++row) {
        for (uint32_t block = 0; block < kOutputBlocksPerRow; ++block) {
            auto* tile_block = narrow_tiles + block * kNarrowTileElements + row * kBlockElements;
            const uint32_t output_index = group_base + (row * kOutputBlocksPerRow + block) * kBlockElements;
#pragma GCC unroll 8
            for (uint32_t lane = 0; lane < kBlockElements; ++lane) {
                const uint32_t local_index = output_index + lane;
                const uint32_t logical_index = source_offset + local_index;
                if constexpr (BoundsChecked) {
                    tile_block[lane] =
                        local_index < output_length && logical_index < source_end ? src[logical_index] : 0.0F;
                } else {
                    tile_block[lane] = src[logical_index];
                }
            }
        }
    }
}

template <bool BoundsChecked>
ALWI void fill_source_narrow_tiles(
    const volatile tt_l1_ptr float* src,
    float* src_tiles01,
    float* src_tiles23,
    const uint32_t source_end,
    const uint32_t source_offset,
    const uint32_t source_left_pad,
    const uint32_t group_base) {
    if constexpr (!BoundsChecked) {
        for (uint32_t block = 0; block < 4; ++block) {
            auto* narrow_tile =
                block < 2 ? src_tiles01 + block * kNarrowTileElements : src_tiles23 + (block - 2) * kNarrowTileElements;
            const uint32_t logical_start = source_offset + group_base + block * kBlockElements - source_left_pad;
            WorkspaceIndexCursor cursor(logical_start);
            for (uint32_t row = 0; row < kRowsPerGroup; ++row) {
                read_workspace_block(src, cursor, narrow_tile + row * kBlockElements);
                // Reading the block already advanced once. Skip the remaining
                // two blocks in this 48 element logical row.
                cursor.advance_block();
                cursor.advance_block();
            }
        }
        return;
    }

    for (uint32_t block = 0; block < 4; ++block) {
        auto* narrow_tile =
            block < 2 ? src_tiles01 + block * kNarrowTileElements : src_tiles23 + (block - 2) * kNarrowTileElements;
        for (uint32_t row = 0; row < kRowsPerGroup; ++row) {
            auto* tile_row = narrow_tile + row * kBlockElements;
            const int32_t logical_start =
                static_cast<int32_t>(source_offset) - static_cast<int32_t>(source_left_pad) +
                static_cast<int32_t>(group_base + (row * kOutputBlocksPerRow + block) * kBlockElements);
            read_workspace_block<BoundsChecked>(src, logical_start, source_end, tile_row);
        }
    }
}

template <bool BoundsChecked>
ALWI void fill_output_narrow_tiles(
    const volatile tt_l1_ptr float* src,
    float* narrow_tiles,
    const uint32_t source_end,
    const uint32_t source_offset,
    const uint32_t output_length,
    const uint32_t group_base) {
    if constexpr (!BoundsChecked) {
        WorkspaceIndexCursor cursor(source_offset + group_base);
        for (uint32_t row = 0; row < kRowsPerGroup; ++row) {
            for (uint32_t block = 0; block < kOutputBlocksPerRow; ++block) {
                auto* tile_block = narrow_tiles + block * kNarrowTileElements + row * kBlockElements;
                read_workspace_block(src, cursor, tile_block);
            }
        }
        return;
    }

    for (uint32_t row = 0; row < kRowsPerGroup; ++row) {
        for (uint32_t block = 0; block < kOutputBlocksPerRow; ++block) {
            auto* tile_block = narrow_tiles + block * kNarrowTileElements + row * kBlockElements;
            const uint32_t output_index = group_base + (row * kOutputBlocksPerRow + block) * kBlockElements;
            const uint32_t available = source_offset < source_end ? source_end - source_offset : 0U;
            const uint32_t logical_end = source_offset + (output_length < available ? output_length : available);
            read_workspace_block<BoundsChecked>(
                src, static_cast<int32_t>(source_offset) + static_cast<int32_t>(output_index), logical_end, tile_block);
        }
    }
}

template <bool TileNative, bool RowMajorNocStaging, bool HybridTileMirror>
ALWI void emit_predict_update_tiles(
    const uint32_t source_addr,
    const uint32_t base_addr,
    const uint32_t cb_src_tile0,
    const uint32_t cb_src_tile1,
    const uint32_t cb_base_tile,
    const uint32_t source_end,
    const uint32_t base_end,
    const uint32_t output_length,
    const uint32_t group_count,
    const uint32_t source_offset,
    const uint32_t base_offset,
    const uint32_t source_left_pad,
    const uint32_t tile_mirror_offset,
    const bool source_tile_mirror,
    const bool base_tile_mirror) {
    CircularBuffer source0_buffer(cb_src_tile0);
    CircularBuffer source1_buffer(cb_src_tile1);
    CircularBuffer base_buffer(cb_base_tile);
    Noc noc;
    const auto* src = reinterpret_cast<volatile tt_l1_ptr float*>(source_addr);
    const auto* base = reinterpret_cast<volatile tt_l1_ptr float*>(base_addr);

    for (uint32_t group = 0; group < group_count; ++group) {
        const uint32_t group_base = group * kGroupOutputElements;
        bool needs_read_barrier = false;
        source0_buffer.reserve_back(2);
        auto* src_tiles01 = reinterpret_cast<float*>(source0_buffer.get_write_ptr());
        source1_buffer.reserve_back(2);
        auto* src_tiles23 = reinterpret_cast<float*>(source1_buffer.get_write_ptr());

        bool source_is_dense = group_base >= source_left_pad;
        uint32_t source_begin = 0;
        if (source_is_dense) {
            const uint32_t group_offset = group_base - source_left_pad;
            source_is_dense = source_offset <= source_end && group_offset <= source_end - source_offset &&
                              kSourcePackedElements <= source_end - source_offset - group_offset;
            if (source_is_dense) {
                source_begin = source_offset + group_offset;
            }
        }
        if constexpr (TileNative) {
            if (source_is_dense && source_begin % kGroupOutputElements == 0) {
                read_aligned_source_group(
                    source_addr, source_begin, source0_buffer.get_write_ptr(), source1_buffer.get_write_ptr());
                needs_read_barrier = true;
            } else if (source_is_dense) {
                fill_source_narrow_tiles<false>(
                    src, src_tiles01, src_tiles23, source_end, source_offset, source_left_pad, group_base);
            } else {
                fill_source_narrow_tiles<true>(
                    src, src_tiles01, src_tiles23, source_end, source_offset, source_left_pad, group_base);
            }
        } else if (source_is_dense) {
            bool loaded_from_tile_mirror = false;
            if constexpr (HybridTileMirror) {
                if (source_tile_mirror && source_begin % kGroupOutputElements == 0) {
                    read_aligned_source_group(
                        source_addr + tile_mirror_offset,
                        source_begin,
                        source0_buffer.get_write_ptr(),
                        source1_buffer.get_write_ptr());
                    loaded_from_tile_mirror = true;
                    needs_read_barrier = true;
                }
            }
            if (!loaded_from_tile_mirror) {
                if constexpr (RowMajorNocStaging) {
                    if (source_begin % kNocL1ReadAlignmentElements == 0) {
                        read_row_major_source_group(
                            source_addr, source_begin, source0_buffer.get_write_ptr(), source1_buffer.get_write_ptr());
                        needs_read_barrier = true;
                    } else {
                        fill_source_row_major<false>(
                            src, src_tiles01, src_tiles23, source_end, source_offset, source_left_pad, group_base);
                    }
                } else {
                    fill_source_row_major<false>(
                        src, src_tiles01, src_tiles23, source_end, source_offset, source_left_pad, group_base);
                }
            }
        } else {
            fill_source_row_major<true>(
                src, src_tiles01, src_tiles23, source_end, source_offset, source_left_pad, group_base);
        }

        base_buffer.reserve_back(3);
        auto* base_tiles = reinterpret_cast<float*>(base_buffer.get_write_ptr());
        const bool base_is_dense = group_base <= output_length && kGroupOutputElements <= output_length - group_base &&
                                   base_offset <= base_end && group_base <= base_end - base_offset &&
                                   kGroupOutputElements <= base_end - base_offset - group_base;
        const uint32_t base_begin = base_offset + group_base;
        if constexpr (TileNative) {
            if (base_is_dense && base_begin % kGroupOutputElements == 0) {
                read_aligned_output_group(base_addr, base_begin, base_buffer.get_write_ptr());
                needs_read_barrier = true;
            } else if (base_is_dense) {
                fill_output_narrow_tiles<false>(base, base_tiles, base_end, base_offset, output_length, group_base);
            } else {
                fill_output_narrow_tiles<true>(base, base_tiles, base_end, base_offset, output_length, group_base);
            }
        } else if (base_is_dense) {
            bool loaded_from_tile_mirror = false;
            if constexpr (HybridTileMirror) {
                if (base_tile_mirror && base_begin % kGroupOutputElements == 0) {
                    read_aligned_output_group(base_addr + tile_mirror_offset, base_begin, base_buffer.get_write_ptr());
                    loaded_from_tile_mirror = true;
                    needs_read_barrier = true;
                }
            }
            if (!loaded_from_tile_mirror) {
                if constexpr (RowMajorNocStaging) {
                    if (base_begin % kNocL1ReadAlignmentElements == 0) {
                        read_row_major_output_group(base_addr, base_begin, base_buffer.get_write_ptr());
                        needs_read_barrier = true;
                    } else {
                        fill_output_row_major<false>(
                            base, base_tiles, base_end, base_offset, output_length, group_base);
                    }
                } else {
                    fill_output_row_major<false>(base, base_tiles, base_end, base_offset, output_length, group_base);
                }
            }
        } else {
            fill_output_row_major<true>(base, base_tiles, base_end, base_offset, output_length, group_base);
        }
        if (needs_read_barrier) {
            noc.async_read_barrier();
        }

        source0_buffer.push_back(2);
        source1_buffer.push_back(2);
        base_buffer.push_back(3);
    }
}

template <bool TileNative, bool RowMajorNocStaging, bool HybridTileMirror>
ALWI void emit_scale_tiles(
    const uint32_t source_addr,
    const uint32_t cb_scale_tile,
    const uint32_t source_end,
    const uint32_t source_offset,
    const uint32_t output_length,
    const uint32_t group_count,
    const uint32_t tile_mirror_offset,
    const bool source_tile_mirror) {
    CircularBuffer scale_buffer(cb_scale_tile);
    Noc noc;
    const auto* src = reinterpret_cast<volatile tt_l1_ptr float*>(source_addr);

    for (uint32_t group = 0; group < group_count; ++group) {
        const uint32_t group_base = group * kGroupOutputElements;
        scale_buffer.reserve_back(3);
        auto* scale_tiles = reinterpret_cast<float*>(scale_buffer.get_write_ptr());
        const bool source_is_dense = group_base <= output_length &&
                                     kGroupOutputElements <= output_length - group_base &&
                                     source_offset <= source_end && group_base <= source_end - source_offset &&
                                     kGroupOutputElements <= source_end - source_offset - group_base;
        const uint32_t source_begin = source_offset + group_base;
        if constexpr (TileNative) {
            if (source_is_dense && source_begin % kGroupOutputElements == 0) {
                read_aligned_output_group(source_addr, source_begin, scale_buffer.get_write_ptr());
                noc.async_read_barrier();
            } else if (source_is_dense) {
                fill_output_narrow_tiles<false>(src, scale_tiles, source_end, source_offset, output_length, group_base);
            } else {
                fill_output_narrow_tiles<true>(src, scale_tiles, source_end, source_offset, output_length, group_base);
            }
        } else if (source_is_dense) {
            bool loaded_from_tile_mirror = false;
            if constexpr (HybridTileMirror) {
                if (source_tile_mirror && source_begin % kGroupOutputElements == 0) {
                    read_aligned_output_group(
                        source_addr + tile_mirror_offset, source_begin, scale_buffer.get_write_ptr());
                    noc.async_read_barrier();
                    loaded_from_tile_mirror = true;
                }
            }
            if (!loaded_from_tile_mirror) {
                if constexpr (RowMajorNocStaging) {
                    if (source_begin % kNocL1ReadAlignmentElements == 0) {
                        read_row_major_output_group(source_addr, source_begin, scale_buffer.get_write_ptr());
                        noc.async_read_barrier();
                    } else {
                        fill_output_row_major<false>(
                            src, scale_tiles, source_end, source_offset, output_length, group_base);
                    }
                } else {
                    fill_output_row_major<false>(
                        src, scale_tiles, source_end, source_offset, output_length, group_base);
                }
            }
        } else {
            fill_output_row_major<true>(src, scale_tiles, source_end, source_offset, output_length, group_base);
        }
        scale_buffer.push_back(3);
    }
}

}  // namespace

void kernel_main() {
    const uint32_t input0_addr = get_arg_val<uint32_t>(0);
    const uint32_t input1_or_length = get_arg_val<uint32_t>(1);
    const uint32_t input_length_or_left_pad = get_arg_val<uint32_t>(2);
    const uint32_t initial_even_slot = get_arg_val<uint32_t>(3);
    const uint32_t initial_odd_slot = get_arg_val<uint32_t>(4);
    const uint32_t chunk_config_addr = get_arg_val<uint32_t>(5);
    const uint32_t route_config_addr = get_arg_val<uint32_t>(6);
    const uint32_t chunk_begin = get_arg_val<uint32_t>(7);
    const uint32_t chunk_count = get_arg_val<uint32_t>(8);
    const uint32_t route_count = get_arg_val<uint32_t>(9);
    const uint32_t tile_mirror_offset = get_arg_val<uint32_t>(10);
    const uint32_t chunks_per_sample = get_arg_val<uint32_t>(11);
    const uint32_t input_pages_per_sample = get_arg_val<uint32_t>(12);

    constexpr uint32_t cb_config = get_compile_time_arg_val(0);
    constexpr uint32_t cb_src_tile0 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_src_tile1 = get_compile_time_arg_val(2);
    constexpr uint32_t cb_base_tile = get_compile_time_arg_val(3);
    constexpr uint32_t cb_input_cache = get_compile_time_arg_val(4);
    constexpr uint32_t cb_sync = get_compile_time_arg_val(5);
    constexpr bool tile_native_workspace = get_compile_time_arg_val(6) != 0;
    constexpr bool inverse = get_compile_time_arg_val(7) != 0;
    constexpr auto boundary_mode = static_cast<ttnn::operations::wavelet::BoundaryMode>(get_compile_time_arg_val(8));
    constexpr uint32_t input_page_size = get_compile_time_arg_val(9);
    constexpr bool row_major_noc_staging = get_compile_time_arg_val(10) != 0;
    constexpr bool hybrid_tile_mirror = get_compile_time_arg_val(11) != 0;
    constexpr uint32_t cb_workspace_a = get_compile_time_arg_val(12);
    constexpr uint32_t cb_workspace_b = get_compile_time_arg_val(13);
    constexpr uint32_t cb_workspace_scratch = get_compile_time_arg_val(14);
    static_assert(
        ttnn::operations::wavelet::is_supported_lwt_boundary_mode(boundary_mode), "Unsupported LWT boundary mode");
    constexpr auto config_args = TensorAccessorArgs<15>();
    constexpr auto input0_args = TensorAccessorArgs<config_args.next_compile_time_args_offset()>();
    constexpr auto input1_args = TensorAccessorArgs<input0_args.next_compile_time_args_offset()>();

    const auto input0 = TensorAccessor(input0_args, input0_addr, input_page_size);
    CircularBuffer config_buffer(cb_config);
    CircularBuffer sync_buffer(cb_sync);
    const uint32_t workspace_a_addr = CircularBuffer(cb_workspace_a).get_write_ptr();
    const uint32_t workspace_b_addr = CircularBuffer(cb_workspace_b).get_write_ptr();
    const uint32_t workspace_scratch_addr = CircularBuffer(cb_workspace_scratch).get_write_ptr();
    const uint32_t initial_even_addr =
        resolve_workspace_slot(initial_even_slot, workspace_a_addr, workspace_b_addr, workspace_scratch_addr);
    const uint32_t initial_odd_addr =
        resolve_workspace_slot(initial_odd_slot, workspace_a_addr, workspace_b_addr, workspace_scratch_addr);

    bool first_local_route = true;
    for (uint32_t local_chunk = 0; local_chunk < chunk_count; ++local_chunk) {
        if (!first_local_route) {
            sync_buffer.wait_front(1);
            sync_buffer.pop_front(1);
        }

        const uint32_t global_work_item = chunk_begin + local_chunk;
        const uint32_t batch_index = global_work_item / chunks_per_sample;
        const uint32_t global_chunk = global_work_item - batch_index * chunks_per_sample;
        const uint32_t input_page = batch_index * input_pages_per_sample;
        const uint32_t* chunk = load_config_page(config_args, chunk_config_addr, cb_config, global_chunk);
        if constexpr (inverse) {
            const auto input1 = TensorAccessor(input1_args, input1_or_length, input_page_size);
            const uint32_t coefficient_length = input_length_or_left_pad;
            const uint32_t approximation_begin = chunk[config_word_index(LwtChunkConfigWord::kIlwtApproximationBegin)];
            const uint32_t approximation_length =
                chunk[config_word_index(LwtChunkConfigWord::kIlwtApproximationLength)];
            const uint32_t detail_begin = chunk[config_word_index(LwtChunkConfigWord::kIlwtDetailBegin)];
            const uint32_t detail_length = chunk[config_word_index(LwtChunkConfigWord::kIlwtDetailLength)];
            config_buffer.pop_front(1);
            initialize_inverse_stream<tile_native_workspace>(
                input0,
                coefficient_length,
                approximation_begin,
                initial_even_addr,
                approximation_length,
                cb_input_cache,
                input_page,
                input_page_size);
            initialize_inverse_stream<tile_native_workspace>(
                input1,
                coefficient_length,
                detail_begin,
                initial_odd_addr,
                detail_length,
                cb_input_cache,
                input_page,
                input_page_size);
        } else {
            const uint32_t input_length = input1_or_length;
            const uint32_t left_pad = input_length_or_left_pad;
            const uint32_t initial_even_begin = chunk[config_word_index(LwtChunkConfigWord::kLwtInitialEvenBegin)];
            const uint32_t initial_even_length = chunk[config_word_index(LwtChunkConfigWord::kLwtInitialEvenLength)];
            const uint32_t initial_odd_begin = chunk[config_word_index(LwtChunkConfigWord::kLwtInitialOddBegin)];
            const uint32_t initial_odd_length = chunk[config_word_index(LwtChunkConfigWord::kLwtInitialOddLength)];
            config_buffer.pop_front(1);
            initialize_lwt_streams<boundary_mode, tile_native_workspace>(
                input0,
                initial_even_addr,
                initial_odd_addr,
                cb_input_cache,
                input_length,
                left_pad,
                initial_even_begin,
                initial_even_length,
                initial_odd_begin,
                initial_odd_length,
                input_page,
                input_page_size);
        }

        for (uint32_t route_index = 0; route_index < route_count; ++route_index) {
            if (route_index > 0) {
                sync_buffer.wait_front(1);
                sync_buffer.pop_front(1);
            }
            const uint32_t config_index = global_chunk * route_count + route_index;
            const uint32_t* route = load_config_page(config_args, route_config_addr, cb_config, config_index);
            const uint32_t route_type = route[config_word_index(RouteConfigWord::kRouteType)];
            const uint32_t source_addr = resolve_workspace_slot(
                route[config_word_index(RouteConfigWord::kRouteSourceAddr)],
                workspace_a_addr,
                workspace_b_addr,
                workspace_scratch_addr);
            const uint32_t source_end = route[config_word_index(RouteConfigWord::kRouteSourceLength)];
            const uint32_t base_addr = resolve_workspace_slot(
                route[config_word_index(RouteConfigWord::kRouteBaseAddr)],
                workspace_a_addr,
                workspace_b_addr,
                workspace_scratch_addr);
            const uint32_t base_end = route[config_word_index(RouteConfigWord::kRouteBaseLength)];
            const uint32_t output_length = route[config_word_index(RouteConfigWord::kRouteOutputLength)];
            const uint32_t source_offset = route[config_word_index(RouteConfigWord::kRouteSourceOffset)];
            const uint32_t base_offset = route[config_word_index(RouteConfigWord::kRouteBaseOffset)];
            const uint32_t source_left_pad = route[config_word_index(RouteConfigWord::kRouteSourceLeftPad)];
            const uint32_t group_count = route[config_word_index(RouteConfigWord::kRouteGroupCount)];
            const uint32_t route_flags = route[config_word_index(RouteConfigWord::kRouteFlags)];
            const bool source_tile_mirror =
                (route_flags & ttnn::operations::wavelet::device_protocol::kRouteFlagSourceTileMirror) != 0;
            const bool base_tile_mirror =
                (route_flags & ttnn::operations::wavelet::device_protocol::kRouteFlagBaseTileMirror) != 0;

            if (route_type == kStepPredict || route_type == kStepUpdate) {
                emit_predict_update_tiles<tile_native_workspace, row_major_noc_staging, hybrid_tile_mirror>(
                    source_addr,
                    base_addr,
                    cb_src_tile0,
                    cb_src_tile1,
                    cb_base_tile,
                    source_end,
                    base_end,
                    output_length,
                    group_count,
                    source_offset,
                    base_offset,
                    source_left_pad,
                    tile_mirror_offset,
                    source_tile_mirror,
                    base_tile_mirror);
            } else if (route_type == kStepScaleEven || route_type == kStepScaleOdd) {
                emit_scale_tiles<tile_native_workspace, row_major_noc_staging, hybrid_tile_mirror>(
                    source_addr,
                    cb_base_tile,
                    source_end,
                    source_offset,
                    output_length,
                    group_count,
                    tile_mirror_offset,
                    source_tile_mirror);
            }
            config_buffer.pop_front(1);
            first_local_route = false;
        }
    }
}
