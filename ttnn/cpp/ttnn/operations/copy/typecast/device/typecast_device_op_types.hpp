// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

struct TypecastParams {
    const tt::tt_metal::DataType input_dtype;
    const tt::tt_metal::DataType output_dtype;
    const tt::tt_metal::MemoryConfig output_memory_config;
    const bool fp32_dest_acc_en = false;
    const bool preserve_fp32_precision = false;
    const bool bfp8_pack_precise = false;
    const std::optional<CoreRangeSet> sub_core_grids = std::nullopt;
};

struct TypecastInputs {
    Tensor input;
    std::optional<Tensor> preallocated_output;
};

// For INT8 tensors holding raw 2's complement bytes, configuring the buffers with tt::DataFormat::Int8
// would corrupt every negative value since it emits sign-magnitude. They are configured as UInt8 instead.
inline tt::DataFormat typecast_buffer_data_format(tt::tt_metal::DataType dtype) {
    return dtype == tt::tt_metal::DataType::INT8 ? tt::DataFormat::UInt8
                                                 : tt::tt_metal::datatype_to_dataformat_converter(dtype);
}

}  // namespace ttnn::prim
