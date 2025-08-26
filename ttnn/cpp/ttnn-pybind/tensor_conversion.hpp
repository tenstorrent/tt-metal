// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

#include "ttnn-pybind/pybind_fwd.hpp"

#include <pybind11/pybind11.h>

namespace tt {

namespace tt_metal {

Tensor convert_python_tensor_to_tt_tensor(
    const pybind11::handle& py_tensor,
    std::optional<DataType> optional_data_type,
    std::optional<Layout> optional_layout,
    const std::optional<Tile>& optional_tile,
    const MemoryConfig& memory_config,
    ttnn::distributed::MeshDevice* device,
    ttnn::QueueId cq_id,
    float pad_value,
    const ttnn::distributed::TensorToMesh* mesh_mapper);
}  // namespace tt_metal
}  // namespace tt
