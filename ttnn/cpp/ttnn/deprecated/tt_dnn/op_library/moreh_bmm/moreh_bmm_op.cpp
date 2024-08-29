// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/op_library/moreh_bmm/moreh_bmm_op.hpp"

#include "ttnn/deprecated/tt_dnn/op_library/moreh_matmul/moreh_matmul_op.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {
namespace operations {
namespace primary {

namespace {
inline void moreh_bmm_validate(const Tensor& input, const Tensor& mat2) {
    const auto& input_shape = input.get_legacy_shape();
    const auto& mat2_shape = mat2.get_legacy_shape();

    TT_ASSERT(
        input.storage_type() == StorageType::DEVICE && mat2.storage_type() == StorageType::DEVICE,
        "input tensors need to be on device");
    TT_ASSERT(input_shape.rank() == 3, "input must be a 3D tensor");
    TT_ASSERT(mat2_shape.rank() == 3, "mat2 must be a 3D tensor");
}

Tensor moreh_bmm_(
    const Tensor& input,
    const Tensor& mat2,
    const std::optional<const Tensor>& output,
    const MemoryConfig& mem_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig> &compute_kernel_config) {
    moreh_bmm_validate(input, mat2);
    return moreh_matmul(input, mat2, false, false, output, std::nullopt, mem_config, compute_kernel_config);
}
}  // namespace

Tensor moreh_bmm(
    const Tensor& input,
    const Tensor& mat2,
    const std::optional<const Tensor> output,
    const MemoryConfig& output_mem_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {

    return moreh_bmm_(input, mat2, output, output_mem_config, compute_kernel_config);
}

}  // namespace primary

}  // namespace operations

}  // namespace tt
