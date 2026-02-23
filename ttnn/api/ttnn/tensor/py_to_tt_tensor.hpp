// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn {

namespace distributed {
class TensorToMesh;
}  // namespace distributed

tt::tt_metal::Tensor convert_python_tensor_to_tt_tensor(
    const tt::tt_metal::Shape& tensor_shape,
    tt::tt_metal::DataType dst_dtype,
    tt::tt_metal::Layout layout,
    const std::optional<tt::tt_metal::Tile>& optional_tile,
    const tt::tt_metal::MemoryConfig& memory_config,
    ttnn::PyDType src_data_type,
    const std::function<tt::tt_metal::HostBuffer(tt::tt_metal::DataType)>& get_host_tensor,
    std::optional<tt::tt_metal::distributed::MeshDevice*> device,
    std::optional<ttnn::QueueId> cq_id,
    const ttnn::distributed::TensorToMesh* mesh_mapper,
    std::optional<float> pad_value = std::nullopt,
    bool col_tilize = false);
}  // namespace ttnn
