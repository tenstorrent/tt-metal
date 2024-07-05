// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "tensor/tensor.hpp"
#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_eager/tensor/host_buffer/functions.hpp"
#include "tt_eager/tensor/tensor_utils.hpp"
#include "tt_eager/tt_dnn/op_library/compute_kernel_config.hpp"
#include "tt_eager/tt_dnn/op_library/run_operation.hpp"
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
