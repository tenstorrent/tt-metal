// SPDX-FileCopyrightText: 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "argmax_common.hpp"
#include "tt-metalium/constants.hpp"
#include "debug/assert.h"
#include "debug/waypoint.h"

constexpr uint32_t face_width = tt::constants::FACE_WIDTH;
constexpr uint32_t face_height = tt::constants::FACE_HEIGHT;
constexpr uint32_t face_size = tt::constants::FACE_HW;

/**
 * @brief Struct container that gathers parameters (e.g., shape, data format)
 * that are related to the input tensor
 */
struct InputContext {
    // Tensor tile shape
    const uint32_t tile_height;
    const uint32_t tile_width;

    // Tensor last two dimensions, in tiles
    const uint32_t input_height;
    const uint32_t input_width;

    // Tensor last two dimensions, logical size
    const uint32_t logical_height;
    const uint32_t logical_width;

    const DataFormat data_format;
    const uint32_t cb_addr;

    // Reminders for calculating padding offsets
    const uint32_t tile_h_rem;
    const uint32_t tile_w_rem;
    const uint32_t face_h_rem;
    const uint32_t face_w_rem;

    // Convenience field to check
    // whether input tensor contains any padding data
    const bool has_padding;

    InputContext() = delete;
    InputContext(const InputContext&) = delete;

    InputContext(
        uint32_t tile_h,
        uint32_t tile_w,
        uint32_t tiles_h,
        uint32_t tiles_w,
        uint32_t data_h,
        uint32_t data_w,
        uint32_t t_h_rem,
        uint32_t t_w_rem,
        uint32_t f_h_rem,
        uint32_t f_w_rem,
        DataFormat format,
        uint32_t l1_cb_addr) :
        tile_height(tile_h),
        tile_width(tile_w),
        input_height(tiles_h),
        input_width(tiles_w),
        logical_height(data_h),
        logical_width(data_w),
        data_format(format),
        cb_addr(l1_cb_addr),
        tile_h_rem(t_h_rem),
        tile_w_rem(t_w_rem),
        face_h_rem(f_h_rem),
        face_w_rem(f_w_rem),
        has_padding((t_h_rem != 0) || (t_w_rem != 0)) {}
};

/**
 * @brief Struct container that gathers parameters related to the output tensor
 */
struct OutputContext {
    uint32_t collected_count;
    uint32_t output_page_id;

    uint32_t* const stack_ptr;
    const uint32_t stack_buffer_size;

    const uint32_t output_cb_addr;
    const uint32_t write_out_count;

    OutputContext() = delete;
    OutputContext(const OutputContext&) = delete;

    OutputContext(uint32_t* ptr, uint32_t size, uint32_t dst_cb_addr, uint32_t out_count, bool keep_dim) :
        collected_count(0),
        output_page_id(0),
        stack_ptr(ptr),
        stack_buffer_size(size),
        output_cb_addr(dst_cb_addr),
        write_out_count(out_count) {}
};

/**
 * @brief Calculates range of valid data (i.e., not padding) within a face of tile
 *
 * @param[out] data_rows Number of rows, within the face, that contain valid data
 * @param[out] data_cols Number of cols, within the face, that contain valid data
 *
 * @param[in] tile_x The x coordinate of the tile this face belongs to
 * @param[in] tile_y The y coordinate of the tile this face belongs to
 * @param[in] face_id The index (0..3) of the face within the tile
 * @param[in] ctx Parameters of the tensor
 *
 */
void get_face_data_range(
    uint32_t& data_rows,
    uint32_t& data_cols,
    uint32_t tile_x,
    uint32_t tile_y,
    uint32_t face_id,
    const InputContext& ctx);

/**
 * @brief Searches for max values and their locations in one tile of the input tensor
 *
 * @tparam DTYPE C++ representation of the input tensor data format
 * @tparam DataFormat Input tensor data format
 *
 * @param[in] ctx Parameters of the input tensor
 * @param[in] tile_x The x coordinate of the tile
 * @param[in] tile_y The y coordinate of the tile
 *
 * @param max_values Array with the maximal values for a range of input tensor rows
 * @param arg_max Array with indices (tensor related) of the maximal values
 * @param[in] max_size The size of max_values and arg_max arrays
 * @param[out] rows_processed Number of tile rows with real (non-padding) data, which
 *             contributed to the output argmax values
 *
 * @note This function is invoked for subsequent tiles in a horizontal pass over the input tensor.
 *       Each call in this pass updates the max_values, arg_max arrays according to the values
 *       found in the tile.
 *       The rows_processed output parameter allows the caller to validate the the number of processed
 *       argmax values for each input tensor, as well as the number of final argmax values obtained
 *       after each horizontal pass.
 */
template <typename DTYPE, DataFormat format>
void process_input_tile(
    const InputContext& ctx,
    uint32_t tile_x,
    uint32_t tile_y,
    DTYPE max_values[],
    uint32_t arg_max[],
    uint32_t max_size,
    uint32_t& rows_processed) {
    const bool has_padding = ctx.has_padding;
    const DataFormat src_data_format = ctx.data_format;
    auto src_ptr = get_tt_l1_ptr_based_on_data_format<format>(ctx.cb_addr);

    rows_processed = 0;

    // Iterate over faces of the tile
    for (uint32_t face_id = 0; face_id < 4; face_id++) {
        uint32_t rows_to_process = face_width;
        uint32_t cols_to_process = face_height;

        // Update for when face intersects the boundary with padding
        if (has_padding) {
            get_face_data_range(rows_to_process, cols_to_process, tile_x, tile_y, face_id, ctx);
            ASSERT(rows_to_process <= face_height);
            ASSERT(cols_to_process <= face_width);
        }

        if (rows_to_process == 0 && cols_to_process == 0) {
            // Face with padding data only
            continue;
        }

        // Account for each tile row only once
        const bool left_side_face = (face_id == 0 || face_id == 2);
        if (left_side_face) {
            rows_processed += rows_to_process;
        }

        // Offset to the face within the tile
        uint32_t face_offset = face_id * face_size;
        volatile tt_l1_ptr DTYPE* face_ptr = src_ptr + face_offset;

        // Go over the rows of the face. Update the maximum values in each row.
        for (uint32_t row = 0; row < rows_to_process; row++) {
            // Row index in the tile.
            const uint32_t row_index = (face_id < 2) ? row : row + face_height;

            ASSERT(row_index < max_size);

            DTYPE curr_max = max_values[row_index];
            uint32_t curr_arg_max = arg_max[row_index];

            // Go over elements in the current row, current face.
            for (uint32_t col = 0; col < cols_to_process; col++) {
                // Index within the face
                uint32_t index = row * face_width + col;

                DTYPE value = face_ptr[index];

                bool new_max = false;
                if constexpr (format == DataFormat::Float16_b) {
                    new_max = bfloat16_greater(value, curr_max);
                } else if constexpr (format == DataFormat::Float32) {
                    new_max = float32_greater(value, curr_max);
                }

                if (new_max) {
                    const bool is_left_side_face = (face_id == 0 || face_id == 2);
                    const uint32_t new_arg_max = tile_x * ctx.tile_width + (is_left_side_face ? 0 : face_width) + col;
                    curr_max = value;
                    curr_arg_max = new_arg_max;
                }
            }
            max_values[row_index] = curr_max;
            arg_max[row_index] = curr_arg_max;
        }
    }
}

/**
 * @brief Stores a sequence of argmax values into a staging area.
 *
 * @tparam keepdim The keepdim parameter of the ttnn argmax call
 *
 * @param[in] new_values Argmax values to be stored
 * @param[in] count Number of the values to be stored
 * @param[in] ctx Parameters related to the output tensor
 *
 * @note The location of where values are stored is managed by the OutputContext object
 */
template <bool keepdim>
void collect_row_major_output(uint32_t new_values[], uint32_t count, OutputContext& ctx) {
    const uint32_t curr_collected = ctx.collected_count;

    if constexpr (keepdim) {
        ASSERT(curr_collected + count <= ctx.stack_buffer_size);
    } else {
        ASSERT(curr_collected + count <= ctx.write_out_count);
    }

    auto* stack_ptr = ctx.stack_ptr;
    auto* cb_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctx.output_cb_addr);

    for (uint32_t idx = 0; idx < count; idx++) {
        uint32_t write_index = curr_collected + idx;
        if constexpr (keepdim) {
            // Accumulate into the on stack array
            stack_ptr[write_index] = new_values[idx];
        } else {
            // Write directly into the output CB
            cb_ptr[write_index] = new_values[idx];
        }
    }

    ctx.collected_count += count;
}

/**
 * @brief Writes argmax values to the output tensor
 *
 * @tparam AccessorType Type of the output tensor TensorAccessor object
 * @tparam keepdim The keepdim parameter of the ttnn argmax call
 *
 * @param[in] output_accessor TensorAccessor of the output tensor
 * @param[in] output_ctx Parameters related to the output tensor
 */
template <typename AccessorType, bool keepdim>
void write_to_output(AccessorType& output_accessor, OutputContext& output_ctx) {
    const uint32_t output_page_elements = output_ctx.write_out_count;
    uint32_t collected_count = output_ctx.collected_count;
    uint32_t output_page_id = output_ctx.output_page_id;

    auto dst_cb_addr = output_ctx.output_cb_addr;

    uint32_t sent_count = 0;
    while (collected_count > 0) {
        // When keepdim is true, argmax values are accumulated in an on-stack buffer.
        // Otherwise, argmax values are accumulated directly in the outut CB.
        if constexpr (keepdim) {
            auto* stack_ptr = output_ctx.stack_ptr;
            auto* dst_cb_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_cb_addr);
            // Copy one page of output data into the output CB.
            for (uint32_t idx = 0; idx < output_page_elements; idx++) {
                dst_cb_ptr[idx] = stack_ptr[sent_count + idx];
            }
        }

        const uint32_t write_size = output_page_elements * sizeof(uint32_t);
        uint64_t dst_noc_addr = get_noc_addr(output_page_id, output_accessor);
        noc_async_write(dst_cb_addr, dst_noc_addr, write_size);

        sent_count += output_page_elements;
        collected_count -= output_page_elements;
        output_page_id++;

        noc_async_write_barrier();
    }

    output_ctx.collected_count = 0;
    output_ctx.output_page_id = output_page_id;
}
