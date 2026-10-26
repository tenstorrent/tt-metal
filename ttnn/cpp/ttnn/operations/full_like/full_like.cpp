// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "full_like.hpp"

#include "ttnn/operations/full/device/full_device_operation.hpp"

namespace ttnn {

Tensor moreh_full_like(
    const Tensor& input,
    const std::variant<float, int> fill_value,
    const std::optional<DataType>& dtype,
    const std::optional<Layout>& layout,
    const std::optional<MemoryConfig>& memory_config) {
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Full Like: Input must be on device");
    const auto& shape = input.logical_shape();

    // prim::full is a creation op with no tensor inputs, so the device-operation framework cannot derive the output's
    // mesh distribution from `input` the way it does for every other op; without help the output would come back
    // fully replicated even for a sharded input. Carry the input's topology through explicitly (as ttnn::empty_like
    // does) so a full_like of a sharded tensor is itself sharded. The output is allocated on every device of the
    // mesh, so only adopt a topology that spans the whole mesh: a sub-mesh distribution would describe shards that
    // the allocation does not have.
    std::optional<tt::tt_metal::TensorTopology> tensor_topology;
    if (input.device()->shape().mesh_size() == input.tensor_topology().distribution_shape().mesh_size()) {
        tensor_topology = input.tensor_topology();
    }

    return ttnn::prim::full(
        ttsl::SmallVector<uint32_t>(shape.cbegin(), shape.cend()),
        fill_value,
        input.device(),
        dtype.value_or(input.dtype()),
        layout.value_or(input.layout()),
        memory_config.value_or(input.memory_config()),
        std::move(tensor_topology));
}

}  // namespace ttnn
