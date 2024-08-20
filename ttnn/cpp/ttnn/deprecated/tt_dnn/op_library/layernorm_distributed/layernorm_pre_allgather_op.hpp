// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"


using namespace tt::constants;

namespace tt {

namespace tt_metal {

enum class LayerNormType {
    LAYERNORM, RMSNORM
};

operation::ProgramWithCallbacks layernorm_pre_allgather_multi_core(
    const Tensor &a,
    Tensor& output,
    LayerNormType norm_type,
    ttnn::DeviceComputeKernelConfig compute_kernel_config);



struct LayerNormPreAllGather {
    LayerNormType norm_type;
    const ttnn::DeviceComputeKernelConfig compute_kernel_config;
    const DataType output_dtype;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
};

}  // namespace metal

namespace operations {

namespace primary {

template <LayerNormType layernorm_type>
struct make_layernorm_pre_allgather {
    Tensor operator()(
        const Tensor& a,
        std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        const DataType output_dtype = DataType::BFLOAT16) const {
        std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({a}))};
        operation::launch_op(
            [compute_kernel_config, output_dtype] (const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
                const auto& a = input_tensors.at(0);
                auto arch = a.storage_type() == StorageType::DEVICE ? a.device()->arch() : AutoFormat::GetDefaultDevice()->arch();
                auto kernel_config_val = init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::HiFi4, false, false, false);
                return operation::run(
                        LayerNormPreAllGather{
                            .norm_type = layernorm_type,
                            .compute_kernel_config = kernel_config_val,
                            .output_dtype = output_dtype},
                        {a});
            }, {a}, output_tensors);
        return output_tensors.at(0);
    }
};

constexpr auto layernorm_pre_allgather = make_layernorm_pre_allgather<LayerNormType::LAYERNORM>{};
constexpr auto rmsnorm_pre_allgather = make_layernorm_pre_allgather<LayerNormType::RMSNORM>{};


}  // namespace primary

}  // namespace operations

}  // namespace tt
