// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/experimental/ccl/strided_all_gather_async/device/strided_all_gather_async_device_operation_types.hpp"

namespace ttnn::operations::experimental::ccl::strided_all_gather_minimal_matmul_async {

struct operation_attributes_t {
    /* All Gather Params */
    const strided_all_gather_async::operation_attributes_t strided_all_gather_async_struct;

    /* Matmul Params */
    const minimal_matmul::MinimalMatmulOp matmul_struct;

    const CoreCoord all_gather_core_grid_offset;
    const bool read_local_slice_from_input;
    const std::vector<IDevice*> devices; // not used, could be deleted?
    const strided_all_gather_async::StridedAllGatherAsync ag_op;
};

struct tensor_args_t {
    const Tensor input_tensor;
    const Tensor weight_tensor;
    const std::optional<Tensor> persistent_output_buffer;
    const std::optional<const Tensor> bias = std::nullopt;
};

using tensor_return_value_t = std::vector<Tensor>;

using spec_return_value_t = std::vector<TensorSpec>;

}  // namespace ttnn::operations::experimental::ccl::strided_all_gather_minimal_matmul_async
