// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <cstdint>

namespace ttnn {
namespace operations {
namespace data_movement {
namespace detail {

tt::tt_metal::Shape infer_dims_for_reshape(const tt::tt_metal::Tensor& tensor, ttsl::Span<const int32_t> shape);

}  // namespace detail
}  // namespace data_movement
}  // namespace operations
}  // namespace ttnn

using PadValue = std::variant<uint32_t, float>;
