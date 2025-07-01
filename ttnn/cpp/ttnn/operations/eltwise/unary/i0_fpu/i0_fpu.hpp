// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {

namespace operations {

namespace unary {

struct I0Fpu {
    static Tensor invoke(
        QueueId queue_id,
        const Tensor& input,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

}  // namespace unary
}  // namespace operations

constexpr auto i0_fpu = ttnn::register_operation<"ttnn::i0_fpu", ttnn::operations::unary::I0Fpu>();

}  // namespace ttnn
