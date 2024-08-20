// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/layernorm_distributed/layernorm_pre_allgather_op.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks layernorm_post_allgather_multi_core(
    const Tensor &a,
    const Tensor &stats,
    const std::optional<const Tensor> gamma,
    const std::optional<const Tensor> beta,
    Tensor& output,
    LayerNormType norm_type,
    float eps,
    ttnn::DeviceComputeKernelConfig compute_kernel_config);



struct LayerNormPostAllGather {
    LayerNormType norm_type;
    float eps;
    MemoryConfig output_mem_config;
    // LayerNormProgramConfig program_config;
    const ttnn::DeviceComputeKernelConfig compute_kernel_config;

    void validate(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor> &output_tensors
    ) const;
};

}  // namespace metal

namespace operations {

namespace primary {

template <LayerNormType layernorm_type>
struct make_layernorm_post_allgather {
    Tensor operator()(
        const Tensor& a,
        const Tensor& stats,
        float eps,
        std::optional<const Tensor> gamma = std::nullopt,
        std::optional<const Tensor> beta = std::nullopt,
        const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        // const LayerNormProgramConfig& program_config = LayerNormDefaultProgramConfig{},
        std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt) const {
        std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({a}))};
        log_debug("layernorm_post_allgather: before launch_op");
        operation::launch_op(
            [eps, mem_config,
            // program_config,
            compute_kernel_config] (const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
                const auto& a = input_tensors.at(0);
                const auto& stats = input_tensors.at(1);
                const auto& gamma = optional_input_tensors.at(0);
                const auto& beta = optional_input_tensors.at(1);
                auto arch = a.storage_type() == StorageType::DEVICE ? a.device()->arch() : AutoFormat::GetDefaultDevice()->arch();
                auto kernel_config_val = init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::HiFi4, false, false, false);
                return operation::run(
                        LayerNormPostAllGather{
                            .norm_type = layernorm_type,
                            .eps = eps,
                            .output_mem_config = mem_config,
                            // .program_config = program_config,
                            .compute_kernel_config = kernel_config_val},
                        {a, stats},
                        {gamma, beta});
            }, {a, stats}, output_tensors, {gamma, beta});
        return output_tensors.at(0);
    }
};

constexpr auto layernorm_post_allgather = make_layernorm_post_allgather<LayerNormType::LAYERNORM>{};
constexpr auto rmsnorm_post_allgather = make_layernorm_post_allgather<LayerNormType::RMSNORM>{};


}  // namespace primary

}  // namespace operations

}  // namespace tt
