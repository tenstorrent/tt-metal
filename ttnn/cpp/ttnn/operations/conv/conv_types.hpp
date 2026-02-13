// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ttnn {
// Enum to specify the type of result to be returned; std::get is used to extract the value from the conv result variants.
//
// The enum values are used to index into the conv result variant typedefs (e.g., Conv2dResultWithOptions and the
// corresponding conv-transpose result variant).
enum class ConvResultType { OUTPUT = 0, OUTPUT_DIM, OUTPUT_WEIGHTS_AND_BIAS, OUTPUT_DIM_WEIGHTS_AND_BIAS };
}  // namespace ttnn
