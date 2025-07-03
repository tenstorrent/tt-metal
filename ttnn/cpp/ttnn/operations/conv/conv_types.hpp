// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ttnn {

namespace operations::conv {
// Enum to specify the type of result to be returned, std::get is used to extract the value from the variant
//
// The enum values are used to index into the ResultWithOptions variant
enum class ResultType { OUTPUT = 0, OUTPUT_DIM, OUTPUT_WEIGHTS_AND_BIAS, OUTPUT_DIM_WEIGHTS_AND_BIAS };
}  // namespace operations::conv
}  // namespace ttnn
