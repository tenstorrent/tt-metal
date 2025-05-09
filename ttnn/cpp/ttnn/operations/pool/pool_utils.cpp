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
    uint32_t kernel_size_h,
    uint32_t kernel_size_w,
    bool ceil_mode,
    uint32_t out_h,
    uint32_t out_w,
    uint32_t stride_h,
    uint32_t stride_w,
    uint32_t ceil_w,
    uint32_t pad_h,
    uint32_t pad_w,
    uint32_t out_stick_x_start,
    uint32_t out_stick_y_start,
    bool count_include_pad,
    std::optional<uint32_t> out_nhw_per_core,
    std::optional<int32_t> divisor_override) {
    std::vector<uint32_t> scalars;
    float value;
    float previous_value = 0.;
    bool first = true;
    bool last_overlaps_padding = false;
    uint32_t last_x = std::numeric_limits<uint32_t>::max();
    uint32_t last_y = std::numeric_limits<uint32_t>::max();

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
                    // value = 1. / (float)(kernel_size_h * kernel_size_w);
                    int hstart = out_stick_y_start * stride_h - pad_h;
                    int wstart = out_stick_x_start * stride_w - pad_w;
                    int hend = std::min((int)(hstart + kernel_size_h), (int)(in_h + ceil_w + pad_h));
                    int wend = std::min((int)(wstart + kernel_size_w), (int)(in_w + ceil_w + pad_w));
                    hstart = std::max(hstart, 0);
                    wstart = std::max(wstart, 0);

                    // Clamp to avoid going beyond real input dimensions (excluding ceil_w-induced padding)
                    int valid_hstart = std::max(hstart, 0);
                    int valid_wstart = std::max(wstart, 0);
                    int valid_hend = std::min(hend, (int)in_h);
                    int valid_wend = std::min(wend, (int)in_w);

                    int pool_area = std::max(0, valid_hend - valid_hstart) * std::max(0, valid_wend - valid_wstart);
                    value = pool_area > 0 ? 1.f / (float)pool_area : 0.f;
                    // if ((in_w - kernel_size_w) % stride_w != 0) {
                    //     if (out_stick_x_start == out_w - 1 && out_stick_y_start == out_h - 1) {
                    //         value = 1. / (float)((kernel_size_h * kernel_size_w) - kernel_size_h * ceil_w -
                    //                              (kernel_size_w - ceil_w) * ceil_w);
                    //     } else if (out_stick_x_start == out_w - 1) {
                    //         value = 1. / (float)((kernel_size_h * kernel_size_w) - kernel_size_w * ceil_w);
                    //     } else if (out_stick_y_start == out_h - 1) {
                    //         value = 1. / (float)((kernel_size_h * kernel_size_w) - kernel_size_w * ceil_w);
                    //     }
                    // }
                    bool on_top_edge = (out_stick_y_start * stride_h - pad_h) < 0;
                    bool on_left_edge = (out_stick_x_start * stride_w - pad_w) < 0;
                    bool on_bottom_edge = (out_stick_y_start * stride_h - pad_h + kernel_size_h) > (int)in_h;
                    bool on_right_edge = (out_stick_x_start * stride_w - pad_w + kernel_size_w) > (int)in_w;

                    bool overlaps_padding = on_top_edge || on_left_edge || on_bottom_edge || on_right_edge;

                    // Use a combination of current tile's (x, y) to track where the kernel layout changes
                    bool is_new_config = first || overlaps_padding != last_overlaps_padding ||
                                         out_stick_y_start != last_y || out_stick_x_start != last_x;

                    if (is_new_config) {
                        scalars.push_back(bfloat16(value).to_packed());
                        first = false;
                        last_overlaps_padding = overlaps_padding;
                        last_y = out_stick_y_start;
                        last_x = out_stick_x_start;
                    }
                    out_stick_x_start = (out_stick_x_start + 1) % out_h;
                    if (out_stick_x_start == 0) {
                        out_stick_y_start = (out_stick_y_start + 1) % out_w;
                    }
                }

            } else {
                value = 1. / (float)(kernel_size_h * kernel_size_w);
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
