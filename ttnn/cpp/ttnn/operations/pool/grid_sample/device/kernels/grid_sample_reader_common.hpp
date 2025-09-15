// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <stdint.h>
#include "dataflow_api.h"

#define ALWI inline __attribute__((always_inline))

// Constants shared between interleaved and sharded kernels
constexpr uint32_t PRECOMPUTED_GRID_ELEMENTS_PER_POINT = 6;
constexpr uint32_t STANDARD_GRID_ELEMENTS_PER_POINT = 2;

// Data type constants (from ttnn/api/ttnn/tensor/types.hpp DataType enum)
constexpr uint32_t DTYPE_BFLOAT16 = 0;
constexpr uint32_t DTYPE_FLOAT32 = 1;

// Utility functions
ALWI bool is_coordinate_valid(int32_t coord, uint32_t max_size) {
    return (coord >= 0) && (coord < static_cast<int32_t>(max_size));
}

ALWI void fill_four_val(uint32_t begin_addr, uint16_t val, uint16_t val1, uint16_t val2, uint16_t val3) {
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(begin_addr);
    ptr[0] = (val | (val1 << 16));
    ptr[1] = (val2 | (val3 << 16));
}

ALWI uint16_t float_to_bfloat16(float value) {
    uint32_t tmp;
    std::memcpy(&tmp, &value, sizeof(tmp));
    return static_cast<uint16_t>(tmp >> 16);
}

ALWI float bfloat16_to_float(uint16_t bf16) {
    uint32_t tmp = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &tmp, sizeof(result));
    return result;
}

// Grid coordinate reading functions
template <uint32_t grid_dtype, bool use_precomputed_grid>
struct GridCoordinateReader {
    // Read grid coordinates and return corner coordinates and weights
    template <typename GridPtrType>
    ALWI static void read_grid_point(
        GridPtrType grid_ptr,
        uint32_t grid_idx,
        float height_scale,
        float height_offset,
        float width_scale,
        float width_offset,
        uint32_t input_height,
        uint32_t input_width,
        int32_t& h0,
        int32_t& h1,
        int32_t& w0,
        int32_t& w1,
        uint16_t& weight_nw_bf,
        uint16_t& weight_ne_bf,
        uint16_t& weight_sw_bf,
        uint16_t& weight_se_bf) {
        if constexpr (use_precomputed_grid) {
            // Each precomputed grid entry has 6 values: h0, w0, weight_nw, weight_ne, weight_sw, weight_se
            const uint32_t precomputed_data_offset = grid_idx * PRECOMPUTED_GRID_ELEMENTS_PER_POINT;
            const int16_t h0_raw = *reinterpret_cast<volatile int16_t*>(&grid_ptr[precomputed_data_offset + 0]);
            const int16_t w0_raw = *reinterpret_cast<volatile int16_t*>(&grid_ptr[precomputed_data_offset + 1]);

            h0 = static_cast<int32_t>(h0_raw);
            w0 = static_cast<int32_t>(w0_raw);
            h1 = h0 + 1;
            w1 = w0 + 1;

            // Read precomputed weights
            weight_nw_bf = grid_ptr[precomputed_data_offset + 2];
            weight_ne_bf = grid_ptr[precomputed_data_offset + 3];
            weight_sw_bf = grid_ptr[precomputed_data_offset + 4];
            weight_se_bf = grid_ptr[precomputed_data_offset + 5];
        } else {
            // Each regular grid entry has 2 values: x, y coordinates
            float h_coord_rel, w_coord_rel;
            if constexpr (grid_dtype == DTYPE_FLOAT32) {
                // For FLOAT32 grid, each coordinate is a 32-bit float
                volatile tt_l1_ptr float* float_data = reinterpret_cast<volatile tt_l1_ptr float*>(grid_ptr);
                const uint32_t float_offset = grid_idx * STANDARD_GRID_ELEMENTS_PER_POINT;
                w_coord_rel = float_data[float_offset + 0];  // x coordinate
                h_coord_rel = float_data[float_offset + 1];  // y coordinate
            } else {
                // For BFLOAT16 grid, read as uint16 and convert
                const uint32_t coordinate_pair_offset = grid_idx * STANDARD_GRID_ELEMENTS_PER_POINT;
                const uint16_t h_coord_raw = grid_ptr[coordinate_pair_offset + 1];  // y coordinate
                const uint16_t w_coord_raw = grid_ptr[coordinate_pair_offset + 0];  // x coordinate
                h_coord_rel = bfloat16_to_float(h_coord_raw);
                w_coord_rel = bfloat16_to_float(w_coord_raw);
            }

            const float h_coord_image = h_coord_rel * height_scale + height_offset;
            const float w_coord_image = w_coord_rel * width_scale + width_offset;

            h0 = static_cast<int32_t>(floor(h_coord_image));
            h1 = h0 + 1;
            w0 = static_cast<int32_t>(floor(w_coord_image));
            w1 = w0 + 1;

            // Calculate bilinear interpolation weights
            const float h0_f = static_cast<float>(h0);
            const float w0_f = static_cast<float>(w0);

            const float h_frac = h_coord_image - h0_f;
            const float w_frac = w_coord_image - w0_f;
            const float h_frac_inv = 1.0f - h_frac;
            const float w_frac_inv = 1.0f - w_frac;

            // Boundary checks
            const bool h0_valid = is_coordinate_valid(h0, input_height);
            const bool h1_valid = is_coordinate_valid(h1, input_height);
            const bool w0_valid = is_coordinate_valid(w0, input_width);
            const bool w1_valid = is_coordinate_valid(w1, input_width);

            const float weight_nw = (h0_valid && w0_valid) ? (h_frac_inv * w_frac_inv) : 0.0f;  // North-West
            const float weight_ne = (h0_valid && w1_valid) ? (h_frac_inv * w_frac) : 0.0f;      // North-East
            const float weight_sw = (h1_valid && w0_valid) ? (h_frac * w_frac_inv) : 0.0f;      // South-West
            const float weight_se = (h1_valid && w1_valid) ? (h_frac * w_frac) : 0.0f;          // South-East

            weight_nw_bf = float_to_bfloat16(weight_nw);
            weight_ne_bf = float_to_bfloat16(weight_ne);
            weight_sw_bf = float_to_bfloat16(weight_sw);
            weight_se_bf = float_to_bfloat16(weight_se);
        }
    }
};

// Input data reading template - handles both tensor accessor and direct NOC reads
template <typename TensorAccessor>
ALWI void read_four_corner_inputs(
    const TensorAccessor& input_tensor_accessor,
    uint32_t batch_offset,
    uint32_t input_width,
    uint32_t input_stick_nbytes,
    int32_t h0,
    int32_t h1,
    int32_t w0,
    int32_t w1,
    uint32_t input_height,
    uint32_t l1_write_input_addr) {
    // Boundary checks (recompute for performance)
    const bool h0_valid = is_coordinate_valid(h0, input_height);
    const bool h1_valid = is_coordinate_valid(h1, input_height);
    const bool w0_valid = is_coordinate_valid(w0, input_width);
    const bool w1_valid = is_coordinate_valid(w1, input_width);

    uint32_t write_addr = l1_write_input_addr;

    // Read 4 corner input sticks via NOC from remote input tensor
    if (h0_valid && w0_valid) {
        const uint32_t north_west_stick_index = batch_offset + (h0 * input_width) + w0;
        const uint64_t remote_noc_addr = input_tensor_accessor.get_noc_addr(north_west_stick_index);
        noc_async_read(remote_noc_addr, write_addr, input_stick_nbytes);
    }
    write_addr += input_stick_nbytes;

    if (h0_valid && w1_valid) {
        const uint32_t north_east_stick_index = batch_offset + (h0 * input_width) + w1;
        const uint64_t remote_noc_addr = input_tensor_accessor.get_noc_addr(north_east_stick_index);
        noc_async_read(remote_noc_addr, write_addr, input_stick_nbytes);
    }
    write_addr += input_stick_nbytes;

    if (h1_valid && w0_valid) {
        const uint32_t south_west_stick_index = batch_offset + (h1 * input_width) + w0;
        const uint64_t remote_noc_addr = input_tensor_accessor.get_noc_addr(south_west_stick_index);
        noc_async_read(remote_noc_addr, write_addr, input_stick_nbytes);
    }
    write_addr += input_stick_nbytes;

    if (h1_valid && w1_valid) {
        const uint32_t south_east_stick_index = batch_offset + (h1 * input_width) + w1;
        const uint64_t remote_noc_addr = input_tensor_accessor.get_noc_addr(south_east_stick_index);
        noc_async_read(remote_noc_addr, write_addr, input_stick_nbytes);
    }
}

// Process single grid point - common logic for both interleaved and sharded
template <
    uint32_t grid_dtype,
    bool use_precomputed_grid,
    uint32_t input_height,
    uint32_t input_width,
    uint32_t input_stick_nbytes,
    uint32_t input_cb_index,
    uint32_t scalar_cb_index,
    typename TensorAccessor,
    typename GridPtrType>
ALWI void process_grid_point(
    GridPtrType grid_ptr, uint32_t grid_idx, const TensorAccessor& input_tensor_accessor, uint32_t batch_offset) {
    // Compute scaling factors as constexpr
    constexpr float input_height_f = float(input_height);
    constexpr float input_width_f = float(input_width);
    constexpr float height_scale = input_height_f * 0.5f;
    constexpr float height_offset = height_scale - 0.5f;
    constexpr float width_scale = input_width_f * 0.5f;
    constexpr float width_offset = width_scale - 0.5f;

    int32_t h0, h1, w0, w1;
    uint16_t weight_nw_bf, weight_ne_bf, weight_sw_bf, weight_se_bf;

    // Read grid coordinates and compute weights
    GridCoordinateReader<grid_dtype, use_precomputed_grid>::template read_grid_point(
        grid_ptr,
        grid_idx,
        height_scale,
        height_offset,
        width_scale,
        width_offset,
        input_height,
        input_width,
        h0,
        h1,
        w0,
        w1,
        weight_nw_bf,
        weight_ne_bf,
        weight_sw_bf,
        weight_se_bf);

    // For precomputed grid, we need to compute boundary checks here
    // since they weren't computed in the coordinate reading phase
    if constexpr (use_precomputed_grid) {
        const bool h0_valid = is_coordinate_valid(h0, input_height);
        const bool h1_valid = is_coordinate_valid(h1, input_height);
        const bool w0_valid = is_coordinate_valid(w0, input_width);
        const bool w1_valid = is_coordinate_valid(w1, input_width);

        // Zero out weights for invalid coordinates
        if (!(h0_valid && w0_valid)) {
            weight_nw_bf = 0;
        }
        if (!(h0_valid && w1_valid)) {
            weight_ne_bf = 0;
        }
        if (!(h1_valid && w0_valid)) {
            weight_sw_bf = 0;
        }
        if (!(h1_valid && w1_valid)) {
            weight_se_bf = 0;
        }
    }

    // Reserve CB space for 4 corner input sticks for this grid point
    cb_reserve_back(input_cb_index, 1);
    const uint32_t l1_write_input_addr = get_write_ptr(input_cb_index);

    // Read 4 corner input sticks
    read_four_corner_inputs(
        input_tensor_accessor,
        batch_offset,
        input_width,
        input_stick_nbytes,
        h0,
        h1,
        w0,
        w1,
        input_height,
        l1_write_input_addr);

    // Store bilinear interpolation weights for this grid point
    cb_reserve_back(scalar_cb_index, 1);
    const uint32_t l1_write_scalar_addr = get_write_ptr(scalar_cb_index);
    fill_four_val(l1_write_scalar_addr, weight_nw_bf, weight_ne_bf, weight_sw_bf, weight_se_bf);
    cb_push_back(scalar_cb_index, 1);

    noc_async_read_barrier();
    cb_push_back(input_cb_index, 1);
}
