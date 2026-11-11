// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>

#include <tt_stl/small_vector.hpp>

#include "ttnn/distributed/tensor_topology.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/memory_config/memory_config.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::full {
struct operation_attributes_t {
    const ttsl::SmallVector<uint32_t> shape;
    const std::variant<float, int> fill_value;
    ttnn::MeshDevice* mesh_device;
    const DataType dtype;
    const Layout layout;
    const MemoryConfig memory_config;
    // How the output is distributed across the mesh. Full has no tensor inputs, so the device-operation framework
    // cannot infer this from anything; when unset the output is allocated fully replicated (the framework default).
    const std::optional<tt::tt_metal::TensorTopology> tensor_topology;

    // Cache key / reflected attributes. `tensor_topology` is left out: it only labels the output tensor (applied when
    // the output is created, not by the program), and TensorTopology has no ttsl::hash support.
    static constexpr auto attribute_names =
        std::forward_as_tuple("mesh_device", "shape", "fill_value", "dtype", "layout", "memory_config");
    auto attribute_values() const {
        return std::forward_as_tuple(mesh_device, shape, fill_value, dtype, layout, memory_config);
    }
};

struct tensor_args_t {};

using spec_return_value_t = tt::tt_metal::TensorSpec;
using tensor_return_value_t = Tensor;
}  // namespace ttnn::operations::full
