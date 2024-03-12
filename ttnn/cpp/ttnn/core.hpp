// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>

#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_eager/tensor/tensor.hpp"
#include "tt_eager/tensor/tensor_impl.hpp"  // TTNN_TENSOR_PRINT_PROFILE
#include "tt_eager/tensor/types.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

namespace core {

inline bool has_storage_type_of(const ttnn::Tensor& tensor, const ttnn::StorageType& storage_type) {
    return tensor.storage_type() == storage_type;
}

inline bool is_tensor_on_device(const ttnn::Tensor& tensor) {
    return tensor.storage_type() == StorageType::DEVICE;
}

inline bool is_tensor_on_multi_device(const ttnn::Tensor& tensor) {
    return tensor.storage_type() == StorageType::MULTI_DEVICE;
}

inline bool is_tensor_on_device_or_multidevice(const ttnn::Tensor& tensor) {
    return is_tensor_on_device(tensor) or is_tensor_on_multi_device(tensor);
}

inline bool any_tensor_on_multi_device(const std::vector<ttnn::Tensor>& tensors) {
    for (const auto& tensor : tensors) {
        if (is_tensor_on_multi_device(tensor)) {
            return true;
        }
    }
    return false;
}

inline void set_printoptions(const std::string& profile) {
    tt::tt_metal::tensor_impl::TTNN_TENSOR_PRINT_PROFILE =
        magic_enum::enum_cast<tt::tt_metal::tensor_impl::TensorPrintProfile>(profile, [](char lhs, char rhs) {
            return std::tolower(lhs) == std::tolower(rhs);
        }).value();
}

}  // namespace core

using core::has_storage_type_of;
using core::is_tensor_on_device;
using core::is_tensor_on_multi_device;
using core::is_tensor_on_device_or_multidevice;
using core::any_tensor_on_multi_device;
using core::set_printoptions;
}  // namespace ttnn
