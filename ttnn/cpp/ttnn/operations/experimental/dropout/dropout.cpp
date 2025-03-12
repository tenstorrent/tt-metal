
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/dropout_device_operation.hpp"
#include "dropout.hpp"
#include "ttnn/attribute_customizer_helpers.hpp"
#include "ttnn/mesh_execution.hpp"

namespace ttnn::operations::experimental {

Tensor DropoutOperation::invoke(
    const Tensor& input_tensor, float prob, float scale, uint32_t seed, bool use_per_device_seed) {
    // If we don't need per-device seeds, call dropout directly
    if (!use_per_device_seed) {
        return ttnn::prim::dropout(input_tensor, prob, scale, seed, DataType::BFLOAT16);
    }

    // Get the automatically created mesh operation adapter type
    using DropoutDeviceOp = dropout::DropoutDeviceOperation;
    using DropoutOp = ttnn::MeshDeviceOperationAdapter<DropoutDeviceOp>;
    using DropoutAttrs = typename DropoutOp::operation_attributes_t;

    return ttnn::launch_mesh_workload<DropoutOp>(
        [seed, prob, scale](const auto& coord, auto* device) -> DropoutAttrs {
            auto seed_offset = device->get_device(coord)->id();
            return DropoutAttrs{
                .output_dtype = DataType::BFLOAT16,
                .output_memory_config = MemoryConfig{},
                .seed = seed + seed_offset,
                .prob = prob,
                .scale = scale};
        },
        ttnn::prim::dropout,
        input_tensor,
        prob,
        scale,
        seed,
        DataType::BFLOAT16);
}

}  // namespace ttnn::operations::experimental
