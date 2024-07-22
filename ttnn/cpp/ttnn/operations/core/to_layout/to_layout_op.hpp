// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "third_party/magic_enum/magic_enum.hpp"
#include "ttnn/experimental/tensor/host_buffer/functions.hpp"
#include "ttnn/experimental/tensor/tensor_utils.hpp"
#include "ttnn/experimental/tt_dnn/op_library/compute_kernel_config.hpp"
#include "ttnn/run_operation.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "ttnn/core.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

namespace operations {

namespace core {

struct ToLayout {
    static Tensor execute_on_worker_thread(
        const ttnn::Tensor& tensor_arg,
        const ttnn::Layout layout,
        const std::optional<ttnn::DataType>& dtype = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        Device* device = nullptr);

    static Tensor execute_on_worker_thread(
        const ttnn::Tensor& tensor_arg,
        const ttnn::Layout layout,
        const std::optional<ttnn::DataType>& dtype = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        DeviceMesh* device = nullptr);
};

}  // namespace core
}  // namespace operations
}  // namespace ttnn
