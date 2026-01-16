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
    operation_attributes_t operation_attributes{.branches = std::move(branches)};

    return ttnn::prim::parallel(operation_attributes);
}

}  // namespace ttnn::operations::experimental::parallel
