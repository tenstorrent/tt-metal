// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <stdint.h>
#include <api/dataflow/dataflow_api.h>
#include "ttnn/operations/pool/device/kernels/pool_kernels_common.hpp"
#include "../grid_sample_reader_common.hpp"

template <
    uint32_t grid_dtype,
    bool use_precomputed_grid,
    bool align_corners,
    uint32_t input_height,
    uint32_t input_width,
    uint32_t input_stick_nbytes,
    typename TensorAccessor,
    typename GridPtrType>
ALWI void process_grid_point_nearest(
    Noc noc,
    GridPtrType grid_ptr,
    uint32_t grid_idx,
    const TensorAccessor& input_tensor_accessor,
    uint32_t batch_offset,
    DataflowBuffer output_dfb,
    uint32_t output_write_offset,
    uint32_t fill_stick_addr) {
    constexpr float input_height_f = float(input_height);
    constexpr float input_width_f = float(input_width);
    constexpr float height_scale =
        align_corners ? ((input_height > 1) ? (input_height_f - 1.0f) * 0.5f : 0.0f) : input_height_f * 0.5f;
    constexpr float width_scale =
        align_corners ? ((input_width > 1) ? (input_width_f - 1.0f) * 0.5f : 0.0f) : input_width_f * 0.5f;
    constexpr float height_offset = align_corners ? 0.0f : -0.5f;
    constexpr float width_offset = align_corners ? 0.0f : -0.5f;

    int32_t nearest_h, nearest_w;
    if constexpr (use_precomputed_grid) {
        const uint32_t precomputed_data_offset = grid_idx * PRECOMPUTED_GRID_ELEMENTS_PER_POINT_NEAREST;
        const int16_t h_raw = *reinterpret_cast<volatile int16_t*>(&grid_ptr[precomputed_data_offset]);
        const int16_t w_raw = *reinterpret_cast<volatile int16_t*>(&grid_ptr[precomputed_data_offset + 1]);
        nearest_h = static_cast<int32_t>(h_raw);
        nearest_w = static_cast<int32_t>(w_raw);
    } else {
        float h_coord_rel, w_coord_rel;
        if constexpr (grid_dtype == DTYPE_FLOAT32) {
            volatile tt_l1_ptr float* float_data = reinterpret_cast<volatile tt_l1_ptr float*>(grid_ptr);
            const uint32_t float_offset = grid_idx * STANDARD_GRID_ELEMENTS_PER_POINT;
            w_coord_rel = float_data[float_offset];
            h_coord_rel = float_data[float_offset + 1];
        } else {
            const uint32_t coordinate_pair_offset = grid_idx * STANDARD_GRID_ELEMENTS_PER_POINT;
            const uint16_t h_coord_raw = grid_ptr[coordinate_pair_offset + 1];
            const uint16_t w_coord_raw = grid_ptr[coordinate_pair_offset];
            h_coord_rel = bf16_to_fp32(h_coord_raw);
            w_coord_rel = bf16_to_fp32(w_coord_raw);
        }

        const float h_coord_image = ((h_coord_rel + 1.0f) * height_scale) + height_offset;
        const float w_coord_image = ((w_coord_rel + 1.0f) * width_scale) + width_offset;
        if constexpr (align_corners) {
            nearest_h = static_cast<int32_t>(round(h_coord_image));
            nearest_w = static_cast<int32_t>(round(w_coord_image));
        } else {
            nearest_h = static_cast<int32_t>(floor(h_coord_image + 0.5f));
            nearest_w = static_cast<int32_t>(floor(w_coord_image + 0.5f));
        }
    }

    // Validate both precomputed and ordinary coordinates. This also rejects padded-shard
    // coordinates and the precomputed -1 sentinel before forming an input page index.
    const bool h_valid = is_coordinate_valid(nearest_h, input_height);
    const bool w_valid = is_coordinate_valid(nearest_w, input_width);
    if (h_valid && w_valid) {
        const uint32_t input_stick_index = batch_offset + (nearest_h * input_width) + nearest_w;
        noc.async_read(
            input_tensor_accessor,
            output_dfb,
            input_stick_nbytes,
            {.page_id = input_stick_index},
            {.offset_bytes = output_write_offset});
    } else {
        UnicastEndpoint self_ep;
        noc.async_read(
            self_ep,
            output_dfb,
            input_stick_nbytes,
            experimental::local_addr(fill_stick_addr, noc.get_noc_id()),
            {.offset_bytes = output_write_offset});
    }
}

template <bool is_sharded>
ALWI void advance_grid_index(
    uint32_t& in_grid_row_idx,
    uint32_t& grid_stick_idx,
    uint32_t& l1_grid_addr,
    uint32_t& grid_points_processed,
    uint32_t& curr_batch,
    const uint32_t grid_batching_factor,
    const uint32_t grid_stick_nbytes,
    const uint32_t grid_hw) {
    ++in_grid_row_idx;
    if (in_grid_row_idx == grid_batching_factor) {
        in_grid_row_idx = 0;
        ++grid_stick_idx;
        if constexpr (is_sharded) {
            l1_grid_addr += grid_stick_nbytes;
        }
        ++grid_points_processed;
        if (grid_points_processed == grid_hw) {
            grid_points_processed = 0;
            ++curr_batch;
        }
    }
}

struct NoGridTensorAccessor {};

template <
    uint32_t grid_dtype,
    bool is_sharded,
    bool use_precomputed_grid,
    bool align_corners,
    uint32_t input_height,
    uint32_t input_width,
    uint32_t input_stick_nbytes,
    uint32_t grid_stick_nbytes,
    uint32_t grid_batching_factor,
    uint32_t grid_hw,
    uint32_t split_reader,
    uint32_t reader_id,
    uint32_t grid_nsticks_per_core,
    uint32_t batch_size,
    uint32_t grid_dfb_id,
    uint32_t output_dfb_id,
    uint32_t fill_dfb_id,
    typename InputTensorAccessor,
    typename GridTensorAccessor>
ALWI void grid_sample_nearest_impl(
    const InputTensorAccessor& input_tensor_accessor,
    const GridTensorAccessor& grid_tensor_accessor,
    uint32_t global_grid_stick_start,
    uint32_t start_page_id) {
    const uint32_t starting_batch = global_grid_stick_start / grid_hw;

    DataflowBuffer grid_dfb(grid_dfb_id);
    DataflowBuffer output_dfb(output_dfb_id);
    DataflowBuffer fill_dfb(fill_dfb_id);
    Noc noc;

    uint32_t grid_stick_idx = 0;
    uint32_t l1_grid_addr = grid_dfb.get_write_ptr();
    const uint32_t fill_stick_addr = fill_dfb.get_write_ptr();
    zero_out_page(noc, fill_dfb);

    uint32_t in_grid_row_idx = 0;
    uint32_t curr_batch = starting_batch;
    uint32_t grid_points_processed = global_grid_stick_start % grid_hw;
    if constexpr (split_reader && reader_id == 1) {
        advance_grid_index<is_sharded>(
            in_grid_row_idx,
            grid_stick_idx,
            l1_grid_addr,
            grid_points_processed,
            curr_batch,
            grid_batching_factor,
            grid_stick_nbytes,
            grid_hw);
    }

    while (grid_stick_idx < grid_nsticks_per_core) {
        volatile tt_l1_ptr uint16_t* grid_stick_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_grid_addr);
        const uint32_t batch_offset = curr_batch * input_height * input_width;

        if constexpr (!is_sharded) {
            noc.async_read(
                grid_tensor_accessor, grid_dfb, grid_stick_nbytes, {.page_id = grid_stick_idx + start_page_id}, {});
            noc.async_read_barrier();
        }
        const uint32_t output_write_offset =
            grid_stick_idx * grid_batching_factor * input_stick_nbytes + in_grid_row_idx * input_stick_nbytes;

        if (curr_batch < batch_size) {
            process_grid_point_nearest<
                grid_dtype,
                use_precomputed_grid,
                align_corners,
                input_height,
                input_width,
                input_stick_nbytes>(
                noc,
                grid_stick_ptr,
                in_grid_row_idx,
                input_tensor_accessor,
                batch_offset,
                output_dfb,
                output_write_offset,
                fill_stick_addr);
        } else {
            UnicastEndpoint self_ep;
            noc.async_read(
                self_ep,
                output_dfb,
                input_stick_nbytes,
                experimental::local_addr(fill_stick_addr, noc.get_noc_id()),
                {.offset_bytes = output_write_offset});
        }

        advance_grid_index<is_sharded>(
            in_grid_row_idx,
            grid_stick_idx,
            l1_grid_addr,
            grid_points_processed,
            curr_batch,
            grid_batching_factor,
            grid_stick_nbytes,
            grid_hw);
        // Each split reader owns alternating grid points, so skip the point assigned to
        // the peer after advancing past the one just processed.
        if constexpr (split_reader) {
            advance_grid_index<is_sharded>(
                in_grid_row_idx,
                grid_stick_idx,
                l1_grid_addr,
                grid_points_processed,
                curr_batch,
                grid_batching_factor,
                grid_stick_nbytes,
                grid_hw);
        }
    }
    noc.async_read_barrier();
}
