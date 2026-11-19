// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/recipe/sdpa_recipe_joint.hpp"
#include "ttnn/operations/transformer/sdpa/device/joint_sdpa_device_operation.hpp"

namespace ttnn::prim {

// Joint SDPA: the precision recipe program with the joint Q/K/V as a second key-stream segment (device/recipe/).
tt::tt_metal::ProgramDescriptor JointSDPADeviceOperation::JointSDPAProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    return ttnn::operations::transformer::sdpa::detail::joint_recipe_descriptor(
        operation_attributes, tensor_args, tensor_return_value);
}

}  // namespace ttnn::prim
