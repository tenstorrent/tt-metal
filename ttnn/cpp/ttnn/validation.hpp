// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>
#include <unordered_set>

#include "ttnn/core.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

namespace validation {

using TensorToValidate = std::variant<ttnn::Tensor, int, float>;

void validate_input_tensor(
    const char* operation_name,
    const std::optional<TensorToValidate>& optional_tensor_to_validate,
    const TensorSchema& input_schema) {
    if (input_schema.is_optional && not optional_tensor_to_validate.has_value()) {
        return;
    }

    const auto& tensor_to_validate = optional_tensor_to_validate.value();

    if (input_schema.can_be_a_scalar) {
        if (std::holds_alternative<int>(tensor_to_validate) or std::holds_alternative<float>(tensor_to_validate)) {
            return;
        }
    }

    if (not std::holds_alternative<ttnn::Tensor>(tensor_to_validate)) {
        TT_THROW("{}: Tensor must be of type ttnn.Tensor, but got {}", operation_name, tensor_to_validate.index());
    }

    const auto& tensor = std::get<ttnn::Tensor>(tensor_to_validate);

    if (tensor.get_shape().rank() < input_schema.min_rank or tensor.get_shape().rank() > input_schema.max_rank) {
        TT_THROW(
            "{}: Tensor rank is not valid: rank is {} but must be  {} <= rank <- {}",
            operation_name,
            tensor.get_shape().rank(),
            input_schema.min_rank,
            input_schema.max_rank);
    }

    if (input_schema.dtypes.find(tensor.get_dtype()) == input_schema.dtypes.end()) {
        TT_THROW("{}: Tensor must be of type {}, but got {}", operation_name, input_schema.dtypes, tensor.get_dtype());
    }

    if (input_schema.layouts.find(tensor.get_layout()) == input_schema.layouts.end()) {
        TT_THROW("{}: Tensor must be of layout {}, but got {}", operation_name, input_schema.layouts, tensor.get_layout());
    }

    if (input_schema.can_be_on_device and input_schema.can_be_on_cpu) {
        // pass
    } else if (input_schema.can_be_on_device) {
        if (not ttnn::is_tensor_on_device_or_multidevice(tensor)) {
            TT_THROW("{}: Tensor must be on device!", operation_name);
        }
    } else if (input_schema.can_be_on_cpu) {
        if (ttnn::has_storage_type_of(tensor, ttnn::DEVICE_STORAGE_TYPE)) {
            TT_THROW("{}: Tensor must be on host!", operation_name);
        }
    } else {
        TT_THROW("{}: Tensor must be on host or device!", operation_name);
    }

    if (not tensor.is_allocated()) {
        TT_THROW("{}: Tensor must be allocated!", operation_name);
    }
}

}  // namespace validation
using validation::validate_input_tensor;
}  // namespace ttnn
