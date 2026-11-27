// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::quasar {

// The dtype the Quasar untilize family emits for `input_dtype`. Block floats and MX formats cannot exist in
// ROW_MAJOR: the unpacker expands them (to Float16_b) and the pack-untilize writes bf16, so they come out as
// BFLOAT16. (data_movement's untilize_output_dtype covers BFLOAT8_B only.)
inline DataType untilize_output_dtype(DataType input_dtype) {
    return (input_dtype == DataType::BFLOAT8_B || tt::tt_metal::is_mx(input_dtype)) ? DataType::BFLOAT16 : input_dtype;
}

Tensor untilize_with_unpadding(
    const Tensor& input_tensor,
    const Shape& output_tensor_end,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    bool use_multicore = true,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

}  // namespace ttnn::operations::experimental::quasar
