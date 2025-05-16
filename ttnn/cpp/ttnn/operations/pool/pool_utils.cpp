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
// For the maxpool it is 1, for the avg pool it is 1/kernel_size or the first scalar in from the vector
uint32_t get_bf16_pool_scalar(
    Pool2DType pool_type,
    uint32_t kernel_h,
    uint32_t kernel_w,
    std::optional<uint32_t> in_h,
    std::optional<uint32_t> in_w,
    std::optional<uint32_t> out_h,
    std::optional<uint32_t> out_w,
    std::optional<uint32_t> stride_h,
    std::optional<uint32_t> stride_w,
    std::optional<bool> ceil_mode,
    std::optional<uint32_t> ceil_w,
    std::optional<uint32_t> out_x,
    std::optional<uint32_t> out_y,
    std::optional<int32_t> divisor_override,
    std::optional<uint32_t> out_nhw_per_core,
    std::vector<uint32_t>* sinchronization_indexes,
    std::vector<uint32_t>* scalars) {
    float value;
    bool first_scalar = true;
    uint32_t packed_first_value;
    uint32_t last_area_signature = 0;
    uint32_t out_x_stick = out_x.value_or(0);
    uint32_t out_y_stick = out_y.value_or(0);

    switch (pool_type) {
        case Pool2DType::MAX_POOL2D:
            value = 1.;
            packed_first_value = bfloat16(value).to_packed();
            if (scalars != nullptr) {
                scalars->push_back(packed_first_value);
            }
            if (sinchronization_indexes != nullptr) {
                sinchronization_indexes->push_back(0);
            }
            break;
        case Pool2DType::AVG_POOL2D:
            if (divisor_override.has_value()) {
                value = 1. / (float)divisor_override.value();
                packed_first_value = bfloat16(value).to_packed();
                if (scalars != nullptr) {
                    scalars->push_back(packed_first_value);
                }
                if (sinchronization_indexes != nullptr) {
                    sinchronization_indexes->push_back(0);
                }
            } else if (ceil_mode.value_or(false) && ceil_w.value_or(0) > 0) {
                for (uint32_t i = 0; i < out_nhw_per_core.value(); i++) {
                    int hstart = out_y_stick * stride_h.value_or(1);
                    int wstart = out_x_stick * stride_w.value_or(1);
                    int hend = ((hstart + (int)kernel_h) < (int)(in_h.value_or(0) + ceil_w.value_or(0)))
                                   ? hstart + (int)kernel_h
                                   : (int)(in_h.value_or(0) + ceil_w.value_or(0));
                    int wend = ((wstart + (int)kernel_w) < (int)(in_w.value_or(0) + ceil_w.value_or(0)))
                                   ? wstart + (int)kernel_w
                                   : (int)(in_w.value_or(0) + ceil_w.value_or(0));

                    // Valid region (input-only, without pad)
                    int valid_hstart = (hstart > 0) ? hstart : 0;
                    int valid_wstart = (wstart > 0) ? wstart : 0;
                    int valid_hend = (hend < (int)in_h.value_or(0)) ? hend : (int)in_h.value_or(0);
                    int valid_wend = (wend < (int)in_w.value_or(0)) ? wend : (int)in_w.value_or(0);

                    int pool_h = valid_hend - valid_hstart;
                    int pool_w = valid_wend - valid_wstart;
                    int pool_area = (pool_h > 0 && pool_w > 0) ? pool_h * pool_w : 0;

                    float value = pool_area > 0 ? 1.f / (float)pool_area : 0.f;

                    uint32_t area_signature = pool_h * pool_w;

                    // Add new scalar if padding config changes
                    if (first_scalar || area_signature != last_area_signature) {
                        if (first_scalar) {
                            packed_first_value = bfloat16(value).to_packed();
                        }
                        if (scalars != nullptr) {
                            scalars->push_back(bfloat16(value).to_packed());
                        }
                        if (sinchronization_indexes != nullptr) {
                            sinchronization_indexes->push_back(i);
                        }
                        first_scalar = false;
                    }
                    last_area_signature = area_signature;

                    out_x_stick = (out_x_stick + 1) % out_h.value_or(0);
                    if (out_x_stick == 0) {
                        out_y_stick = (out_y_stick + 1) % out_w.value_or(0);
                    }
                }
            } else {
                value = 1. / (float)(kernel_h * kernel_w);
                packed_first_value = bfloat16(value).to_packed();
                if (scalars != nullptr) {
                    scalars->push_back(packed_first_value);
                }
                if (sinchronization_indexes != nullptr) {
                    sinchronization_indexes->push_back(0);
                }
            }
            break;
        default: TT_FATAL(false, "Unsupported pool operation type");
    }
    return packed_first_value;
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
