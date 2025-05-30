// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "pool_utils.hpp"
#include <limits>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/assert.hpp>

namespace ttnn::operations::pool {
// This function generates a vector of elements of type ScalarInfo. It is called once per core and generates the
// adequate scalars for each output element that core should produce. It should be called only for avg pool operation
// and only if the divisor_override is NOT set and the idea behind it is to generate config tesnor in cases where one
// scalar per core is not sufficient to create correct result. Those scenarios are ceil_mode == true and (ceil_pad_h > 0
// || ceil_pad_w > 0) or count_include_pad == false || (pad_h > 0 || pad_w > 0). Both of these scenarios can be
// irrelevant if the divisor_override is set, in which case we don't calculate the divisor since it is already passed as
// an argument. It only adds scalars that are different than the scalar preeceding it not to have duplicates of data,
// this is why we use start and end indices to know how many sequential output elements should be multiplied by the same
// scalar value.
std::vector<ScalarInfo> get_bf16_avg_pool_config_scalars(
    AvgPoolConfig config, uint32_t output_stick_x, uint32_t output_stick_y) {
    std::vector<ScalarInfo> scalars;
    float value;
    bool first_scalar = true;
    uint32_t last_pool_area = 0;

    if ((config.ceil_mode && (config.ceil_w > 0 || config.ceil_h > 0)) ||
        (!config.count_include_pad && (config.pad_h > 0 || config.pad_w > 0))) {
        for (uint32_t i = 0; i < config.out_nhw_per_core; i++) {
            // Compute starting and ending indices of the pooling window
            int h_start = output_stick_x * config.stride_h - config.pad_h;
            int w_start = output_stick_y * config.stride_w - config.pad_w;
            int h_end = std::min(
                h_start + static_cast<int>(config.kernel_h),
                static_cast<int>(config.in_h + config.pad_h + config.ceil_h));
            int w_end = std::min(
                w_start + static_cast<int>(config.kernel_w),
                static_cast<int>(config.in_w + config.pad_w + config.ceil_w));

            int valid_h_start = (h_start > 0) ? h_start : 0;
            int valid_w_start = (w_start > 0) ? w_start : 0;
            int valid_h_end = std::min(h_end, static_cast<int>(config.in_h));
            int valid_w_end = std::min(w_end, static_cast<int>(config.in_w));

            int effective_h = valid_h_end - valid_h_start;
            int effective_w = valid_w_end - valid_w_start;
            int pool_area = 0;
            if (config.count_include_pad) {
                // Initial pool area
                pool_area = (h_end - h_start) * (w_end - w_start);

                // Calculate ceil induced padding overflow beyond input dimensions
                int pad_h_over = std::max(h_end - static_cast<int>(config.in_h) - static_cast<int>(config.pad_h), 0);
                int pad_w_over = std::max(w_end - static_cast<int>(config.in_w) - static_cast<int>(config.pad_w), 0);

                // Adjust pool area to exclude padded overflow
                pool_area -= pad_h_over * config.kernel_w;
                pool_area -= pad_w_over * config.kernel_h;

                // Re-add intersection if both directions overflowed
                if (pad_h_over > 0 && pad_w_over > 0) {
                    pool_area += pad_h_over * pad_w_over;
                }

                // Avoid division by zero
                pool_area = pool_area > 1 ? pool_area : 1;
            } else {
                pool_area = (effective_h > 0 && effective_w > 0) ? effective_h * effective_w : 0;
            }
            float value = pool_area > 0 ? 1.f / (float)pool_area : 0.f;

            // Add new scalar if padding config changes
            if (first_scalar || (uint32_t)pool_area != last_pool_area) {
                if (!scalars.empty()) {
                    scalars.back().end = i;
                }
                scalars.push_back({i, bfloat16(value).to_packed() << 16, i});
                first_scalar = false;
            }
            last_pool_area = (uint32_t)pool_area;

            // Advance output element coordinates
            output_stick_y = (output_stick_y + 1) % config.out_w;
            if (output_stick_y == 0) {
                output_stick_x = (output_stick_x + 1) % config.out_h;
            }
        }
    } else {
        value = 1. / (float)(config.kernel_h * config.kernel_w);
        scalars.push_back({0, bfloat16(value).to_packed() << 16, config.out_nhw_per_core});
    }
    scalars.back().end = config.out_nhw_per_core;
    return scalars;
}

// Return a single bf16 scalar for the pool type in u32 (packed in the least 16 bits)
// For the maxpool it is 1, for the avg pool it is 1/kernel_size or the divisor override used to initialize compile
// time argument sent to kernels. If there are multiple scalars needed call get_bf16_avg_pool_config_scalars
uint32_t get_bf16_pool_scalar(
    Pool2DType pool_type, uint32_t kernel_h, uint32_t kernel_w, std::optional<int32_t> divisor_override) {
    float value;

    switch (pool_type) {
        case Pool2DType::MAX_POOL2D: value = 1.; break;
        case Pool2DType::AVG_POOL2D:
            if (divisor_override.has_value()) {
                value = 1. / (float)divisor_override.value();
            } else {
                value = 1. / (float)(kernel_h * kernel_w);
            }
            break;
        default: TT_FATAL(false, "Unsupported pool operation type");
    }
    return bfloat16(value).to_packed() << 16;
}

// Return a single bf16 init value for the pool type in u32 (packed in the least 16 bits)
uint32_t get_bf16_pool_init_value(Pool2DType pool_type) {
    float value;
    switch (pool_type) {
        case Pool2DType::MAX_POOL2D: value = -std::numeric_limits<float>::infinity(); break;
        case Pool2DType::AVG_POOL2D: value = 0.; break;
        default: TT_FATAL(false, "Unsupported pool operation type");
    }
    return bfloat16(value).to_packed();
}

std::map<std::string, std::string> get_defines(Pool2DType pool_type) {
    std::map<std::string, std::string> defines;
    switch (pool_type) {
        case Pool2DType::MAX_POOL2D: defines["REDUCE_OP"] = "PoolType::MAX"; break;
        case Pool2DType::AVG_POOL2D: defines["REDUCE_OP"] = "PoolType::SUM"; break;
        default: TT_FATAL(false, "Unsupported pool operation type");
    }
    defines["REDUCE_DIM"] = "ReduceDim::REDUCE_COL";

    return defines;
}
}  // namespace ttnn::operations::pool
