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

using TensorToValidate = std::variant<std::optional<ttnn::Tensor>, int, float>;
using TensorsToValidate = std::vector<TensorToValidate>;

inline void validate_input_tensor(
    const std::string& operation_name, const TensorToValidate& tensor_to_validate, const TensorSchema& schema) {
    if (schema.can_be_scalar) {
        if (std::holds_alternative<int>(tensor_to_validate) or std::holds_alternative<float>(tensor_to_validate)) {
            return;
        }
    } else {
        if (not std::holds_alternative<std::optional<ttnn::Tensor>>(tensor_to_validate)) {
            TT_THROW("{}: Tensor cannot be a scalar!", operation_name);
        }
    }

    const auto& optional_tensor = std::get<std::optional<ttnn::Tensor>>(tensor_to_validate);

    if (schema.is_optional && not optional_tensor.has_value()) {
        return;
    }

    const auto& tensor = optional_tensor.value();

    if (tensor.get_shape().rank() < schema.min_rank or tensor.get_shape().rank() > schema.max_rank) {
        TT_THROW(
            "{}: Tensor rank is not valid: rank is {} but must be  {} <= rank <- {}",
            operation_name,
            tensor.get_shape().rank(),
            schema.min_rank,
            schema.max_rank);
    }

    if (schema.dtypes.find(tensor.get_dtype()) == schema.dtypes.end()) {
        TT_THROW("{}: Tensor must be of type {}, but got {}", operation_name, schema.dtypes, tensor.get_dtype());
    }

    if (schema.layouts.find(tensor.get_layout()) == schema.layouts.end()) {
        TT_THROW("{}: Tensor must be of layout {}, but got {}", operation_name, schema.layouts, tensor.get_layout());
    }

    if (schema.can_be_on_device and schema.can_be_on_cpu) {
        // pass
    } else if (schema.can_be_on_device) {
        if (not ttnn::is_tensor_on_device_or_multidevice(tensor)) {
            TT_THROW("{}: Tensor must be on device!", operation_name);
        }
    } else if (schema.can_be_on_cpu) {
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

template <typename TensorSchemas>
inline void validate_input_tensors(
    const std::string& operation_name, const TensorsToValidate& tensors, const TensorSchemas& schemas) {
    if (tensors.size() != schemas.size()) {
        TT_THROW(
            "{}: Number of tensors ({}) does not match the number of schemas ({})",
            operation_name,
            tensors.size(),
            schemas.size());
    }
    for (auto index = 0; index < tensors.size(); index++) {
        const auto& tensor = tensors.at(index);
        validate_input_tensor(operation_name, tensor, schemas.at(index));
    }
}

}  // namespace validation
using validation::validate_input_tensor;
using validation::validate_input_tensors;
}  // namespace ttnn
