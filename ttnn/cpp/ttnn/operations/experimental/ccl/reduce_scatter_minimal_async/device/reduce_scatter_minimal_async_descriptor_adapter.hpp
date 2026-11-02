// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim::detail {

// Type bundle for MeshDeviceOperationAdapter<...>::DescriptorMeshWorkloadAdapter, shared by the Ring and Line
// factories (the device operation itself cannot be named from the factory headers it includes).
struct ReduceScatterMinimalAsyncDescriptorAdapterOperation {
    using operation_attributes_t = ReduceScatterMinimalAsyncParams;
    using tensor_args_t = ReduceScatterMinimalAsyncInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
};

}  // namespace ttnn::experimental::prim::detail
