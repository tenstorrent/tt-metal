// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

#include <optional>

namespace ttnn::prim {

enum class AccumulationOp : uint8_t { CUMSUM, CUMPROD };

struct AccumulationParams {
    const int32_t dim;
    const DataType dtype;
    const MemoryConfig output_memory_config;
    const bool flip;
    const AccumulationOp op;
    // When true, skip the compensated (Kahan) accumulation that fp32 cumsum uses by default and
    // fall back to the plain sequential sum. An escape hatch for parity checks and debugging; it
    // has no effect on cumprod or on integer formats, which never take the compensated path.
    const bool disable_compensation = false;
};

struct AccumulationInputs {
    const Tensor& input_tensor;
    std::optional<Tensor> opt_output;
};

}  // namespace ttnn::prim
