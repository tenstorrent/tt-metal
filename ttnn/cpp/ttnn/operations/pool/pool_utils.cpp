// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pool_utils.hpp"
#include <limits>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/assert.hpp>
#include <algorithm>

namespace ttnn::operations::pool {
// Return a single bf16 scalar for the pool type in u32 (packed in the least 16 bits)
std::vector<uint32_t> get_bf16_pool_scalar(
    Pool2DType pool_type,
    uint32_t in_h,
    uint32_t in_w,
    uint32_t kernel_h,
    uint32_t kernel_w,
    bool ceil_mode,
    uint32_t out_h,
    uint32_t out_w,
    uint32_t stride_h,
    uint32_t stride_w,
    uint32_t ceil_w,
    uint32_t pad_h,
    uint32_t pad_w,
    uint32_t out_x,
    uint32_t out_y,
    bool count_include_pad,
    std::vector<uint32_t>& sinchronization_indexes,
    std::optional<uint32_t> out_nhw_per_core,
    std::optional<int32_t> divisor_override) {
    std::vector<uint32_t> scalars;
    float value;
    float previous_value = 0.;
    bool first_scalar = true;
    uint32_t last_area_signature = 0;

    switch (pool_type) {
        case Pool2DType::MAX_POOL2D:
            value = 1.;
            scalars.push_back(bfloat16(value).to_packed());
            break;
        case Pool2DType::AVG_POOL2D:
            if (divisor_override.has_value()) {
                value = 1. / (float)divisor_override.value();
                scalars.push_back(bfloat16(value).to_packed());
            } else if ((ceil_mode && ceil_w > 0) || (!count_include_pad && (pad_h > 0 || pad_w > 0))) {
                for (uint32_t i = 0; i < out_nhw_per_core.value(); i++) {
                    int hstart = out_y * stride_h - pad_h;
                    int wstart = out_x * stride_w - pad_w;
                    int hend = ((hstart + (int)kernel_h) < (int)(in_h + pad_h + ceil_w)) ? hstart + (int)kernel_h
                                                                                         : (int)(in_h + pad_h + ceil_w);
                    int wend = ((wstart + (int)kernel_w) < (int)(in_w + pad_w + ceil_w)) ? wstart + (int)kernel_w
                                                                                         : (int)(in_w + pad_w + ceil_w);

                    // Valid region (input-only, without pad)
                    int valid_hstart = (hstart > 0) ? hstart : 0;
                    int valid_wstart = (wstart > 0) ? wstart : 0;
                    int valid_hend = (hend < (int)in_h) ? hend : (int)in_h;
                    int valid_wend = (wend < (int)in_w) ? wend : (int)in_w;

                    int pool_h = valid_hend - valid_hstart;
                    int pool_w = valid_wend - valid_wstart;
                    int pool_area = (pool_h > 0 && pool_w > 0) ? pool_h * pool_w : 0;

                    float value = pool_area > 0 ? 1.f / (float)pool_area : 0.f;

                    uint32_t area_signature = pool_h * pool_w;

                    // Add new scalar if padding config changes
                    if (first_scalar || area_signature != last_area_signature) {
                        scalars.push_back(bfloat16(value).to_packed());
                        sinchronization_indexes.push_back(i);
                        first_scalar = false;
                    }
                    last_area_signature = area_signature;

                    out_x = (out_x + 1) % out_h;
                    if (out_x == 0) {
                        out_y = (out_y + 1) % out_w;
                    }
                }

            } else {
                value = 1. / (float)(kernel_h * kernel_w);
                scalars.push_back(bfloat16(value).to_packed());
            }
            break;
        default: TT_FATAL(false, "Unsupported pool operation type");
    }
    return scalars;
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
