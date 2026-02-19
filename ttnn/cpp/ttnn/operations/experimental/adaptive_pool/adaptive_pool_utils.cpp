// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "adaptive_pool_utils.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <tt_stl/assert.hpp>

namespace ttnn::operations::experimental::adaptive_pool {

// Kernels dimension (width or height) can be only uniform and give correct result by padding the border elements
// Therefore for feasibility of making kernels uniform we need to make sure that the middle elements already are
bool are_middle_kernels_uniform(const std::vector<uint32_t>& kernels) {
    if (kernels.size() <= 2) {
        return true;
    }

    // Check all middle elements (excluding first and last)
    uint32_t expected_middle = kernels[1];
    for (size_t i = 1; i < kernels.size() - 1; i++) {
        if (kernels[i] != expected_middle) {
            return false;
        }
    }
    return true;
}

// Helper function designed to generate the kernel sizes for each output elements the same way it is
// calculated in pytorch implementation
std::pair<std::vector<uint32_t>, std::vector<uint32_t>> calculate_actual_kernel_patterns(
    uint32_t input_h, uint32_t input_w, uint32_t output_h, uint32_t output_w) {
    std::vector<uint32_t> h_kernels, w_kernels;

    // Height kernel pattern
    for (uint32_t out_h = 0; out_h < output_h; out_h++) {
        uint32_t start_h = (out_h * input_h) / output_h;
        uint32_t end_h = ((out_h + 1) * input_h + output_h - 1) / output_h;
        h_kernels.push_back(end_h - start_h);
    }

    // Width kernel pattern
    for (uint32_t out_w = 0; out_w < output_w; out_w++) {
        uint32_t start_w = (out_w * input_w) / output_w;
        uint32_t end_w = ((out_w + 1) * input_w + output_w - 1) / output_w;
        w_kernels.push_back(end_w - start_w);
    }

    return {h_kernels, w_kernels};
}

// Helper function designed to generate the stride values for each output elements the same way it is
// calculated in pytorch implementation
std::vector<uint32_t> calculate_actual_stride_patterns(
    uint32_t /*input_size*/,
    uint32_t output_size,
    uint32_t /*kernel_size*/,
    uint32_t stride,
    uint32_t /*pad_before*/,
    uint32_t /*pad_after*/) {
    std::vector<uint32_t> actual_strides;

    // Calculate the actual start positions in the padded input for each output
    for (uint32_t out_idx = 1; out_idx < output_size; out_idx++) {
        uint32_t prev_start = (out_idx - 1) * stride;
        uint32_t curr_start = out_idx * stride;
        actual_strides.push_back(curr_start - prev_start);
    }

    return actual_strides;
}

// Adaptive pool params carry the information which is needed by pool2d but not given as arguments of the adaptive pool
// op This function calculates the kernel sizes the same way they would be produced by pytorch, analyzes the possibility
// to make them uniform with padding if possible and needed, padds the input tesnor and then for the padded tensor
// calculates strides to check if those would be uniform as well
AdaptivePoolingParams calculate_adaptive_pool_params(
    uint32_t input_h, uint32_t input_w, uint32_t output_h, uint32_t output_w) {
    AdaptivePoolingParams params = {};

    // Get kernel patterns
    auto [h_kernels, w_kernels] = calculate_actual_kernel_patterns(input_h, input_w, output_h, output_w);

    // Calculate basic uniform parameters
    uint32_t base_stride_h = input_h / output_h;
    uint32_t base_stride_w = input_w / output_w;
    uint32_t base_kernel_h = (input_h + output_h - 1) / output_h;  // ceil
    uint32_t base_kernel_w = (input_w + output_w - 1) / output_w;  // ceil

    bool h_uniform = are_middle_kernels_uniform(h_kernels) && !h_kernels.empty() &&
                     std::all_of(h_kernels.begin(), h_kernels.end(), [&](uint32_t k) { return k == h_kernels[0]; });
    bool w_uniform = are_middle_kernels_uniform(w_kernels) && !w_kernels.empty() &&
                     std::all_of(w_kernels.begin(), w_kernels.end(), [&](uint32_t k) { return k == w_kernels[0]; });

    if (h_uniform && w_uniform) {
        params.kernel_size = {base_kernel_h, base_kernel_w};
        params.stride = {base_stride_h, base_stride_w};
        params.padding = {0, 0, 0, 0};
    } else {
        uint32_t target_kernel_h = h_kernels.size() >= 3 ? h_kernels[1] : base_kernel_h;
        uint32_t target_kernel_w = w_kernels.size() >= 3 ? w_kernels[1] : base_kernel_w;

        params.kernel_size = {target_kernel_h, target_kernel_w};

        uint32_t optimized_stride_h = base_stride_h;
        uint32_t optimized_stride_w = base_stride_w;

        uint32_t pad_top = 0, pad_bottom = 0;
        if (h_kernels.size() >= 3) {
            uint32_t first_h = h_kernels[0];
            uint32_t last_h = h_kernels[h_kernels.size() - 1];
            uint32_t middle_h = h_kernels[1];

            if (first_h < middle_h || last_h < middle_h) {
                optimized_stride_h = middle_h - 1;

                if (first_h < middle_h) {
                    pad_top = middle_h - first_h;
                }

                if (last_h < middle_h) {
                    pad_bottom = middle_h - last_h;
                }
            }
        }

        uint32_t pad_left = 0, pad_right = 0;
        if (w_kernels.size() >= 3) {
            uint32_t first_w = w_kernels[0];
            uint32_t last_w = w_kernels[w_kernels.size() - 1];
            uint32_t middle_w = w_kernels[1];

            if (first_w < middle_w || last_w < middle_w) {
                optimized_stride_w = middle_w - 1;

                if (first_w < middle_w) {
                    pad_left = middle_w - first_w;
                }

                if (last_w < middle_w) {
                    pad_right = middle_w - last_w;
                }
            }
        }

        params.stride = {optimized_stride_h, optimized_stride_w};
        params.padding = {pad_top, pad_bottom, pad_left, pad_right};
    }

    uint32_t padded_h = input_h + params.padding[0] + params.padding[1];
    uint32_t padded_w = input_w + params.padding[2] + params.padding[3];
    uint32_t final_out_h = ((padded_h - params.kernel_size[0]) / params.stride[0]) + 1;
    uint32_t final_out_w = ((padded_w - params.kernel_size[1]) / params.stride[1]) + 1;

    if (final_out_h != output_h || final_out_w != output_w) {
        params.kernel_size = {base_kernel_h, base_kernel_w};
        params.stride = {base_stride_h, base_stride_w};
        params.padding = {0, 0, 0, 0};
    }

    return params;
}

// Check if border elements are smaller than or equal to middle (required for padding correction)
bool are_borders_correctable_with_padding(const std::vector<uint32_t>& kernels) {
    if (kernels.size() <= 2) {
        return true;
    }

    uint32_t first = kernels[0];
    uint32_t last = kernels[kernels.size() - 1];
    uint32_t middle_value = kernels[1];

    // Border elements must be <= middle for padding to work
    return (first <= middle_value) && (last <= middle_value);
}

// Check if the calculated pooling parameters produce uniform behavior
bool validate_pooling_params_uniformity(
    const AdaptivePoolingParams& params, uint32_t input_h, uint32_t input_w, uint32_t output_h, uint32_t output_w) {
    // Check height dimension
    auto h_actual_strides = calculate_actual_stride_patterns(
        input_h, output_h, params.kernel_size[0], params.stride[0], params.padding[0], params.padding[1]);

    // Check width dimension
    auto w_actual_strides = calculate_actual_stride_patterns(
        input_w, output_w, params.kernel_size[1], params.stride[1], params.padding[2], params.padding[3]);

    // All strides should be exactly the same (since we're using fixed stride)
    for (uint32_t stride : h_actual_strides) {
        if (stride != params.stride[0]) {
            return false;
        }
    }

    for (uint32_t stride : w_actual_strides) {
        if (stride != params.stride[1]) {
            return false;
        }
    }

    return true;
}

// Validation function to check if adaptive pooling approach is feasible
void validate_adaptive_pool_feasibility(uint32_t input_h, uint32_t input_w, uint32_t output_h, uint32_t output_w) {
    auto [h_kernels, w_kernels] = calculate_actual_kernel_patterns(input_h, input_w, output_h, output_w);

    // Check kernel pattern correctability
    bool h_uniform_middle = are_middle_kernels_uniform(h_kernels);
    bool h_borders_ok = are_borders_correctable_with_padding(h_kernels);
    bool h_correctable = h_uniform_middle && h_borders_ok;

    bool w_uniform_middle = are_middle_kernels_uniform(w_kernels);
    bool w_borders_ok = are_borders_correctable_with_padding(w_kernels);
    bool w_correctable = w_uniform_middle && w_borders_ok;

    // Calculate the actual pooling parameters that would be used
    AdaptivePoolingParams params = calculate_adaptive_pool_params(input_h, input_w, output_h, output_w);

    // Check if the calculated parameters produce uniform behavior
    bool params_uniform = validate_pooling_params_uniformity(params, input_h, input_w, output_h, output_w);

    if (!h_correctable || !w_correctable || !params_uniform) {
        std::string error_msg = "Adaptive pooling configuration not supported. ";

        if (!h_correctable) {
            error_msg += "Height kernel pattern [";
            for (size_t i = 0; i < h_kernels.size(); i++) {
                error_msg += std::to_string(h_kernels[i]);
                if (i < h_kernels.size() - 1) {
                    error_msg += ",";
                }
            }
            error_msg += "] - ";
            if (!h_uniform_middle) {
                error_msg += "middle elements not uniform, ";
            }
            if (!h_borders_ok) {
                error_msg += "border elements larger than middle (padding won't help), ";
            }
        }

        if (!w_correctable) {
            error_msg += "Width kernel pattern [";
            for (size_t i = 0; i < w_kernels.size(); i++) {
                error_msg += std::to_string(w_kernels[i]);
                if (i < w_kernels.size() - 1) {
                    error_msg += ",";
                }
            }
            error_msg += "] - ";
            if (!w_uniform_middle) {
                error_msg += "middle elements not uniform, ";
            }
            if (!w_borders_ok) {
                error_msg += "border elements larger than middle (padding won't help), ";
            }
        }

        if (!params_uniform) {
            error_msg += "Calculated pooling parameters (kernel=[" + std::to_string(params.kernel_size[0]) + "," +
                         std::to_string(params.kernel_size[1]) + "], stride=[" + std::to_string(params.stride[0]) +
                         "," + std::to_string(params.stride[1]) + "], padding=[" + std::to_string(params.padding[0]) +
                         "," + std::to_string(params.padding[1]) + "," + std::to_string(params.padding[2]) + "," +
                         std::to_string(params.padding[3]) + "]) do not produce uniform pooling behavior, ";
        }

        error_msg +=
            "Correctable patterns require uniform middle kernels AND borders <= middle AND calculated pooling "
            "parameters must produce uniform behavior. ";
        error_msg += "Input shape: (" + std::to_string(input_h) + "," + std::to_string(input_w) + "), Output shape: (" +
                     std::to_string(output_h) + "," + std::to_string(output_w) + ").";

        TT_THROW("{}", error_msg);
    }
}

}  // namespace ttnn::operations::experimental::adaptive_pool
