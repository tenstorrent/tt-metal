// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "utils/bfloat16.h"
#include "utils/float32.h"
#include "utils/int32.h"
#include "utils/uint32.h"

#include <cstdint>

template <DataFormat data_format>
auto get_default_value() {
    // Check for supported datatypes
    if constexpr (data_format == DataFormat::Float16_b) {
        return uint16_t{NEG_INF_BFLOAT16};
    } else if constexpr (data_format == DataFormat::Float32) {
        return uint32_t{NEG_INF_FLOAT32};
    } else if constexpr (data_format == DataFormat::Int32) {
        return int32_t{NEG_INF_INT32};
    } else if constexpr (data_format == DataFormat::UInt32) {
        return uint32_t{MIN_UINT32};
    } else {
        // Add other data formats as needed
        static_assert(data_format != data_format, "Unsupported data format");
    }
}

template <DataFormat data_format>
void compare_values(
    const uint32_t src_cb_addr,
    decltype(get_default_value<data_format>())& max_val,
    uint32_t& max_idx,
    const uint32_t i,
    const uint32_t j,
    const uint32_t k,
    const uint32_t red_dim_units,
    bool reduce_all,
    uint32_t inner_dim_units) {
    if constexpr (data_format == DataFormat::Float16_b) {
        volatile tt_l1_ptr uint16_t* in_vals = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(src_cb_addr);
        uint16_t val = in_vals[i];
        if (bfloat16_greater(val, max_val)) {
            max_idx = reduce_all ? (k * inner_dim_units * red_dim_units + j * red_dim_units + i) : i;
            max_val = val;
        }
    } else if constexpr (data_format == DataFormat::Float32) {
        volatile tt_l1_ptr uint32_t* in_vals = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_cb_addr);
        uint32_t val = in_vals[i];
        if (float32_greater(val, max_val)) {
            max_idx = reduce_all ? (k * inner_dim_units * red_dim_units + j * red_dim_units + i) : i;
            max_val = val;
        }
    } else if constexpr (data_format == DataFormat::Int32) {
        volatile tt_l1_ptr int32_t* in_vals = reinterpret_cast<volatile tt_l1_ptr int32_t*>(src_cb_addr);
        int32_t val = in_vals[i];
        if (int32_greater(val, max_val)) {
            max_idx = reduce_all ? (k * inner_dim_units * red_dim_units + j * red_dim_units + i) : i;
            max_val = val;
        }
    } else if constexpr (data_format == DataFormat::UInt32) {
        volatile tt_l1_ptr uint32_t* in_vals = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_cb_addr);
        uint32_t val = in_vals[i];
        if (uint32_greater(val, max_val)) {
            max_idx = reduce_all ? (k * inner_dim_units * red_dim_units + j * red_dim_units + i) : i;
            max_val = val;
        }
    } else {
        static_assert(data_format != data_format, "Unsupported data format in compare_values");
    }
}
