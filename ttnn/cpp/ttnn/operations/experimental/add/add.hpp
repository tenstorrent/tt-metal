// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

#include "ttnn/operations/experimental/add/device/add_device_operation.hpp"

#include <optional>

namespace ttnn::operations::experimental::binary {

struct AddOperation {
    static Tensor invoke(
        const Tensor& a,
        const Tensor& b,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> output_tensor = std::nullopt) {
        if (output_dtype.has_value() && output_tensor.has_value()) {
            TT_FATAL(
                output_dtype.value() == output_tensor.value().dtype(),
                "Both output dtype and output tensor provided dtype should match");
        }

        auto [operation_attributes, tensor_args] = ttnn::experimental::prim::AddDeviceOperation::invoke(
            a, b, output_dtype, memory_config, std::move(output_tensor));
        return ttnn::device_operation::launch<ttnn::experimental::prim::AddDeviceOperation>(
            operation_attributes, tensor_args);
    }
};

constexpr auto add = ttnn::register_operation<"ttnn::experimental::add", AddOperation>();
}  // namespace ttnn::operations::experimental::binary
