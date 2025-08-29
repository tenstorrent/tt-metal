// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

#include "ttnn-pybind/pybind_fwd.hpp"
#include <optional>
#include <string>

namespace tt {

namespace tt_metal {

struct PyTensorPreparedConversion {
    /// Use this layout to construct the initial tensor -- extra conversion might be done
    /// after the tensor has been moved to device.
    Layout construct_with_layout = Layout::TILE;
    DataType construct_with_data_type = DataType::INVALID;
    std::optional<std::string> torch_convert_dtype = std::nullopt;
};

std::optional<DataType> map_torch_data_type_to_ttnn(const std::string& py_dtype);

Tensor create_tt_tensor_from_py_data(
    std::size_t py_data_ptr,
    const Shape& py_data_shape,
    const TensorLayout& tensor_layout,
    ttnn::distributed::MeshDevice* device,
    const tt::tt_metal::MemoryPin& pydata_pin,
    ttnn::QueueId cq_id,
    float pad_value,
    const ttnn::distributed::TensorToMesh* mesh_mapper);

std::optional<PyTensorPreparedConversion> prepare_torch_tensor_conversion(
    const std::string& torch_dtype,
    bool is_tensor_empty,
    const std::optional<DataType>& dtype,
    const std::optional<Layout>& layout,
    bool has_device,
    const MemoryConfig& memory_config,
    const std::optional<Tile>& optional_tile);

}  // namespace tt_metal
}  // namespace tt
