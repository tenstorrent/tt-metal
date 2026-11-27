// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "whole_padding_tiles.hpp"

#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"

namespace ttnn::operations::normalization::detail {

Tensor on_default_tile_grid(const Tensor& tensor) {
    if (tensor.layout() != tt::tt_metal::Layout::TILE || tensor.storage_type() != StorageType::DEVICE ||
        tensor.is_sharded()) {
        return tensor;
    }
    const auto default_padded_shape =
        tt::tt_metal::TensorSpec(
            tensor.logical_shape(),
            tt::tt_metal::TensorLayout(
                tensor.dtype(), tensor.tensor_spec().page_config(), tt::tt_metal::MemoryConfig{}))
            .padded_shape();
    if (tensor.padded_shape() == default_padded_shape) {
        return tensor;
    }
    return ttnn::reshape(tensor, tensor.logical_shape());
}

}  // namespace ttnn::operations::normalization::detail
