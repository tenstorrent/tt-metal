// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "parallel.hpp"
#include "device/parallel_device_operation.hpp"

namespace ttnn::operations::experimental {

ParallelReturnType ExecuteParallel::invoke(std::vector<BranchDescriptor> branches) {
    return invoke_impl(std::move(branches));
}

ParallelReturnType ExecuteParallel::invoke_impl(std::vector<BranchDescriptor> branches) {
    TT_FATAL(!branches.empty(), "ParallelDeviceOperation requires at least one branch");

    // Extract mesh device from the first input tensor of the first branch
    ttnn::MeshDevice* mesh_device = nullptr;
    for (const auto& branch : branches) {
        auto input_tensors = branch.get_input_tensors();
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

    ttnn::experimental::prim::ParallelParams operation_attributes{
        .branches = std::move(branches), .mesh_device = mesh_device};

    return ttnn::prim::parallel(std::move(operation_attributes));
}

}  // namespace ttnn::operations::experimental
