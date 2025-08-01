// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/run_operation.hpp"
#include <tt-metalium/command_queue.hpp>
#include "ttnn/core.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

namespace operations {

namespace core {

struct ToLayout {
    static Tensor invoke(
        const ttnn::Tensor& tensor_arg,
        ttnn::Layout layout,
        const std::optional<ttnn::DataType>& dtype = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace core
}  // namespace operations
}  // namespace ttnn
