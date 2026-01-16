// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "parallel.hpp"
#include "device/parallel_device_operation.hpp"

namespace ttnn::operations::experimental::parallel {

tensor_return_value_t ExecuteParallel::invoke(std::vector<std::shared_ptr<BranchDescriptor>> branches) {
    return invoke_impl(std::move(branches));
}

tensor_return_value_t ExecuteParallel::invoke_impl(std::vector<std::shared_ptr<BranchDescriptor>> branches) {
    TT_FATAL(!branches.empty(), "ParallelDeviceOperation requires at least one branch");

    // Extract mesh device from the first input tensor of the first branch
    ttnn::MeshDevice* mesh_device = nullptr;
    for (const auto& branch : branches) {
        auto input_tensors = branch->get_input_tensors();
        for (const auto* tensor : input_tensors) {
            if (tensor != nullptr && tensor->storage_type() == StorageType::DEVICE) {
                mesh_device = tensor->device();
                break;
            }
        }
        if (mesh_device != nullptr) {
            break;
        }
    }
    TT_FATAL(mesh_device != nullptr, "Could not find mesh device from input tensors");

    operation_attributes_t operation_attributes{.branches = std::move(branches), .mesh_device = mesh_device};

    return ttnn::prim::parallel(operation_attributes);
}

}  // namespace ttnn::operations::experimental::parallel
