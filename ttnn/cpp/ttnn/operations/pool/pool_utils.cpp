// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "pool_utils.hpp"
#include <limits>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/assert.hpp>
#include <algorithm>

namespace ttnn::operations::pool {
//
std::vector<ScalarInfo> get_bf16_avg_pool_config_scalars(
    Pool2DType pool_type,
    uint32_t kernel_h,
    uint32_t kernel_w,
    uint32_t in_h,
    uint32_t in_w,
    uint32_t out_h,
    uint32_t out_w,
    uint32_t stride_h,
    uint32_t stride_w,
    bool ceil_mode,
    uint32_t ceil_h,
    uint32_t ceil_w,
    uint32_t out_x,
    uint32_t out_y,
    uint32_t pad_h,
    uint32_t pad_w,
    uint32_t out_nhw_per_core,
    std::optional<int32_t> divisor_override) {
    std::vector<ScalarInfo> scalars;
    float value;
    bool first_scalar = true;
    uint32_t last_pool_area = 0;

    switch (pool_type) {
        case Pool2DType::MAX_POOL2D:
            value = 1.;
            scalars.push_back({0, bfloat16(value).to_packed() << 16, out_nhw_per_core - 1});

            break;
        case Pool2DType::AVG_POOL2D:
            if (divisor_override.has_value()) {
                value = 1. / (float)divisor_override.value();
                scalars.push_back({0, bfloat16(value).to_packed(), out_nhw_per_core - 1});

            } else if (ceil_mode && (ceil_w > 0 || ceil_h > 0)) {
                for (uint32_t i = 0; i < out_nhw_per_core; i++) {
                    // Initial kernel window start based on stride and padding
                    int hstart = out_x * stride_h - pad_h;
                    int wstart = out_y * stride_w - pad_w;
                    int hend = hstart + kernel_h;
                    int wend = wstart + kernel_w;

                    int pool_area;

                    // Count how many *actual* kernel elements fall within the padded input bounds
                    pool_area = (hend - hstart) * (wend - wstart);
                    pool_area -= ((hend > (int)in_h) ? (hend - in_h) : 0) * kernel_w;
                    pool_area -= ((wend > (int)in_w) ? (wend - in_w) : 0) * kernel_h;
                    // Remove doubly subtracted corner if both overflows happened
                    if (hend > (int)in_h && wend > (int)in_w) {
                        pool_area += (hend - in_h) * (wend - in_w);
                    }
                    pool_area = std::max(1, pool_area);  // Avoid division by zero

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

                    out_y = (out_y + 1) % out_w;
                    if (out_y == 0) {
                        out_x = (out_x + 1) % out_h;
                    }
                }
            } else {
                value = 1. / (float)(kernel_h * kernel_w);
                scalars.push_back({0, bfloat16(value).to_packed() << 16, out_nhw_per_core - 1});
            }
            break;
        default: TT_FATAL(false, "Unsupported pool operation type");
    }
    scalars.back().end = out_nhw_per_core;
    return scalars;
}

// Return a single bf16 scalar for the pool type in u32 (packed in the least 16 bits)
// For the maxpool it is 1, for the avg pool it is 1/kernel_size or the divisor override used to initialize compile time
// argument sent to kernels. If there are multiple scalars needed call get_bf16_avg_pool_config_scalars
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
