// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "batch_norm_utils.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt_stl/assert.hpp>

using namespace tt::tt_metal;

namespace ttnn::operations::normalization::batch_norm::utils {

DeviceComputeKernelConfig resolve_compute_kernel_config(
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config, const Tensor& input) {
    TT_FATAL(
        input.storage_type() == StorageType::DEVICE,
        "Invalid input tensor storage type: Input tensor must be on device. (storage type={})",
        input.storage_type());

    const auto arch = input.device()->arch();
    const auto input_data_format = datatype_to_dataformat_converter(input.dtype());
    const auto default_math_fidelity = MathFidelity::HiFi4;
    const auto default_approx_mode = false;
    const auto default_fp32_acc = input_data_format == tt::DataFormat::UInt32 ||
                                  input_data_format == tt::DataFormat::Int32 ||
                                  input_data_format == tt::DataFormat::Float32;
    const auto default_l1_acc = true;
    const auto default_dst_full_sync_en = false;
    return init_device_compute_kernel_config(
        arch,
        compute_kernel_config,
        default_math_fidelity,
        default_approx_mode,
        default_fp32_acc,
        default_l1_acc,
        default_dst_full_sync_en);
}

}  // namespace ttnn::operations::normalization::batch_norm::utils
