// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>

#include "ttnn/types.hpp"

namespace ttnn {

namespace core {

inline bool has_storage_type_of(const ttnn::Tensor& tensor, const ttnn::StorageType& storage_type) {
    return tensor.storage_type() == storage_type;
}

}  // namespace core

using core::has_storage_type_of;
}  // namespace ttnn
