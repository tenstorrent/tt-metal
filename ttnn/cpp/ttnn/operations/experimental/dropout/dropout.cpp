
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/dropout_device_operation.hpp"
#include "dropout.hpp"
#include "ttnn/attribute_customizer_helpers.hpp"

namespace ttnn::operations::experimental {

Tensor DropoutOperation::invoke(
    const Tensor& input_tensor, float prob, float scale, uint32_t seed, bool use_per_device_seed) {
    // Create a scoped customizer that only modifies the seed if use_per_device_seed is true
    auto customizer_guard = ttnn::with_customizer<dropout::DropoutMeshDeviceOperation>(
        [use_per_device_seed](const auto& attrs, const auto& coord, auto* device) {
            if (use_per_device_seed) {
                auto device_attrs = attrs;
                device_attrs.seed = attrs.seed + device->get_device(coord)->id();
                return device_attrs;
            }
            return attrs;
        });

    // Single call path - customizer is only active if use_per_device_seed was true
    return ttnn::prim::dropout(input_tensor, prob, scale, seed, DataType::BFLOAT16);
}

}  // namespace ttnn::operations::experimental
