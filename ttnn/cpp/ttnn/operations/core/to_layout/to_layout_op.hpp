// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <magic_enum/magic_enum.hpp>
#include <tt-metalium/command_queue.hpp>
#include <functional>
#include <optional>

#include "ttnn/core.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

namespace tt {
namespace tt_metal {
enum class Layout;
}  // namespace tt_metal
}  // namespace tt

namespace ttnn {

namespace operations {

namespace core {

struct ToLayout {
    static Tensor invoke(
        const ttnn::Tensor& tensor_arg,
        const ttnn::Layout layout,
        const std::optional<ttnn::DataType>& dtype = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        IDevice* device = nullptr);

    static Tensor invoke(
        const ttnn::Tensor& tensor_arg,
        const ttnn::Layout layout,
        const std::optional<ttnn::DataType>& dtype = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        MeshDevice* device = nullptr);
};

}  // namespace core
}  // namespace operations
}  // namespace ttnn
