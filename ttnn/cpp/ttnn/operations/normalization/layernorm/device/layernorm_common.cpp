// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/normalization/layernorm/device/layernorm_common.hpp"
#include <tt-metalium/host_api.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

std::pair<std::optional<tt::tt_metal::Tensor>, uint32_t> create_reciprocal_tensor_if_needed(
    tt::tt_metal::IDevice* device, uint32_t W, const tt::tt_metal::CoreRangeSet& cores, const bool use_welford) {
    const auto num_cores = cores.num_cores();
    std::optional<tt::tt_metal::Tensor> recip_tensor = std::nullopt;
    uint32_t reciprocal_CB_size_bytes = 0;
    if (use_welford) {
        const auto recip_dtype = tt::tt_metal::DataType::FLOAT32;
        const tt::tt_metal::ShardSpec shard_spec(cores, {1, W}, tt::tt_metal::ShardOrientation::ROW_MAJOR);
        const tt::tt_metal::MemoryConfig mem_config = tt::tt_metal::MemoryConfig{
            tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED, tt::tt_metal::BufferType::L1, shard_spec};
        const tt::tt_metal::TensorLayout tensor_layout(
            tt::tt_metal::TensorLayout(recip_dtype, tt::tt_metal::Layout::ROW_MAJOR, mem_config));
        const tt::tt_metal::Shape tensor_shape{num_cores, W};
        const tt::tt_metal::TensorSpec tensor_spec(tensor_shape, tensor_layout);
        // Compute the reciprocals of an ascending sequence of integers
        std::vector<float> reciprocals(num_cores * W);
        for (uint32_t i = 0; i < W; i++) {
            // Compute for first row
            reciprocals[i] = 1.0f / (i + 1);
        }
        for (uint32_t i = 1; i < num_cores; i++) {
            // Copy to other rows
            std::copy(reciprocals.begin(), reciprocals.begin() + W, reciprocals.begin() + i * W);
        }

        if (auto* p_mesh_device = dynamic_cast<tt::tt_metal::distributed::MeshDevice*>(device)) {
            recip_tensor = tt::tt_metal::Tensor::from_vector(std::move(reciprocals), tensor_spec, p_mesh_device);
        } else {
            TT_THROW("Cannot cast to MeshDevice");
        }

        reciprocal_CB_size_bytes = recip_tensor->buffer()->aligned_size_per_bank();
    }

    return std::make_pair(recip_tensor, reciprocal_CB_size_bytes);
}

}  // namespace ttnn::prim
