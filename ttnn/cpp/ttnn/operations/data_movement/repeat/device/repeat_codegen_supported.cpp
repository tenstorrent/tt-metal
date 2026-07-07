// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/repeat/device/repeat_codegen_supported.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/types.hpp"

namespace ttnn::prim {

bool supported_by_codegen(const ttnn::Tensor& tensor, int32_t repeat_dim) {
    if (tensor.memory_config().is_sharded()) {
        return false;
    }
    if (tensor.layout() != ttnn::TILE_LAYOUT) {
        return false;
    }
    if (tensor.dtype() != tt::tt_metal::DataType::BFLOAT16) {
        return false;
    }
    const auto& shape = tensor.logical_shape();
    if (shape.rank() != 4) {
        return false;
    }
    if (repeat_dim < 0 || repeat_dim >= 3) {
        return false;
    }
    if (shape[2] % tt::constants::TILE_HEIGHT != 0 || shape[3] % tt::constants::TILE_WIDTH != 0) {
        return false;
    }
    return true;
}

}  // namespace ttnn::prim

namespace ttnn::operations::data_movement::repeat {

ImplementationSelector parse_implementation(const std::optional<std::string>& value) {
    if (!value.has_value() || *value == "auto") {
        return ImplementationSelector::Auto;
    }
    if (*value == "native") {
        return ImplementationSelector::Native;
    }
    if (*value == "codegen") {
        return ImplementationSelector::Codegen;
    }
    TT_FATAL(false, "repeat: invalid implementation \"{}\" (expected \"auto\", \"native\", or \"codegen\")", *value);
}

}  // namespace ttnn::operations::data_movement::repeat
