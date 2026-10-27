// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/experimental/reduction/fast_reduce_nc/fast_reduce_nc.hpp"
#include "ttnn/operations/experimental/reduction/fast_reduce_nc/device/fast_reduce_nc_device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::reduction {

namespace {
// Same default as reduce_op: fp32 DEST accumulation, HiFi3 on Wormhole (HiFi4 + fp32 DEST is unreliable there).
ttnn::DeviceComputeKernelConfig default_kernel_config(
    tt::ARCH arch, const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    auto config = compute_kernel_config.value_or(init_device_compute_kernel_config(
        arch,
        std::nullopt,
        arch == tt::ARCH::WORMHOLE_B0 ? tt::tt_metal::MathFidelity::HiFi3 : tt::tt_metal::MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true));
    verify_numerical_configuration(arch, compute_kernel_config);
    return config;
}
}  // namespace

ttnn::Tensor fast_reduce_nc(
    const ttnn::Tensor& input,
    ttsl::Span<const int32_t> dims,
    const std::optional<const Tensor>& output,
    const ttnn::MemoryConfig& memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids,
    const std::optional<tt::tt_metal::DataType>& output_dtype,
    bool fp32_intermediate_stages,
    bool fast_and_approximate_mode) {
    TT_FATAL(
        input.storage_type() == StorageType::DEVICE,
        "Input tensor storage type must be DEVICE but got {}",
        input.storage_type());

    TT_FATAL(!dims.empty(), "fast_reduce_nc dims should not be empty");

    const auto arch = input.device()->arch();
    const auto kernel_config_val = default_kernel_config(arch, compute_kernel_config);

    ttsl::SmallVector<int32_t> sorted_dims(dims.begin(), dims.end());
    std::sort(sorted_dims.begin(), sorted_dims.end());

    const std::optional<tt::tt_metal::DataType> intermediate_dtype =
        fp32_intermediate_stages ? std::optional<tt::tt_metal::DataType>{tt::tt_metal::DataType::FLOAT32}
                                 : std::nullopt;
    // Accurate fp32 sums on the SFPU (the FPU truncates fp32 to tf32); falls back to the FPU like reduce_op.
    const auto use_sfpu_reduce = [&](const Tensor& stage_input) {
        return !fast_and_approximate_mode && stage_input.dtype() == tt::tt_metal::DataType::FLOAT32 &&
               arch != tt::ARCH::QUASAR && kernel_config_val.fp32_dest_acc_en;
    };
    auto temp_input = input;
    for (uint32_t i = dims.size() - 1; i > 0; i--) {
        auto temp_output = ttnn::prim::fast_reduce_nc(
            temp_input,
            sorted_dims[i],
            std::nullopt,
            memory_config,
            kernel_config_val,
            sub_core_grids,
            intermediate_dtype,
            use_sfpu_reduce(temp_input));
        temp_input = temp_output;
    }
    return ttnn::prim::fast_reduce_nc(
        temp_input,
        sorted_dims.front(),
        output,
        memory_config,
        kernel_config_val,
        sub_core_grids,
        output_dtype,
        use_sfpu_reduce(temp_input));
}

std::tuple<Tensor, Tensor> fast_reduce_nc_split(
    const Tensor& input,
    int32_t dim,
    uint32_t split_output_width,
    const MemoryConfig& memory_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Split reduction requires device input");
    const auto config = default_kernel_config(input.device()->arch(), compute_kernel_config);
    auto outputs = ttnn::prim::fast_reduce_nc_split(input, dim, split_output_width, memory_config, config);
    return {outputs.at(0), outputs.at(1)};
}

}  // namespace ttnn::experimental::reduction
