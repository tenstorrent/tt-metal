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

inline const std::optional<MemoryConfig> get_memory_config(const ttnn::Tensor& tensor) {
    if (not tensor.is_allocated()) {
        return std::nullopt;
    }
    return std::visit(
        [](const auto& storage) -> std::optional<MemoryConfig> {
            using T = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<T, DeviceStorage>) {
                try {
                    return storage.memory_config();
                } catch (...) {
                    return std::nullopt;
                }
            } else if constexpr (std::is_same_v<T, MultiDeviceStorage>) {
                try {
                    return storage.memory_config();
                } catch (...) {
                    return std::nullopt;
                }
            } else {
                return std::nullopt;
            }
        },
        tensor.get_storage());
}

inline void set_printoptions(const std::string& profile) {
    tt::tt_metal::tensor_impl::TTNN_TENSOR_PRINT_PROFILE =
        magic_enum::enum_cast<tt::tt_metal::tensor_impl::TensorPrintProfile>(profile, [](char lhs, char rhs) {
            return std::tolower(lhs) == std::tolower(rhs);
        }).value();
}

}  // namespace core

using core::get_memory_config;
using core::has_storage_type_of;
using core::set_printoptions;
using tt::tt_metal::any_tensor_on_multi_device;
using tt::tt_metal::is_tensor_on_device;
using tt::tt_metal::is_tensor_on_device_or_multidevice;
using tt::tt_metal::is_tensor_on_multi_device;
}  // namespace ttnn
