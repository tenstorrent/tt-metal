// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ttnn::operations::conv {
// Enum to specify the type of result to be returned, std::get is used to extract the value from the variant
//
// The enum values are used to index into the ResultWithOptions variant
enum class ResultType : std::uint8_t { OUTPUT = 0, OUTPUT_DIM, OUTPUT_WEIGHTS_AND_BIAS, OUTPUT_DIM_WEIGHTS_AND_BIAS };
}  // namespace ttnn::operations::conv
