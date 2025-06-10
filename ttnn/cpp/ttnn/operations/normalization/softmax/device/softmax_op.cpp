// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_op.hpp"
#include <tt-metalium/assert.hpp>
#include <tt-metalium/base_types.hpp>
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/run_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/util.hpp>

#include <optional>
#include <type_traits>

using uint32_t = std::uint32_t;
using namespace tt::constants;

namespace ttnn::operations::normalization {

void Softmax::validate(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    using tt::tt_metal::DataType;
    using tt::tt_metal::Layout;
    using tt::tt_metal::StorageType;

    TT_FATAL(input_tensors.size() == 1 and optional_input_tensors.size() <= 1, "Must have 1 or 2 input tensors");
    auto& input_tensor = input_tensors.at(0);
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to softmax need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to softmax need to be allocated in buffers on device!");
    TT_FATAL((input_tensor.layout() == Layout::TILE), "Inputs to softmax must be tilized");
    TT_FATAL(
        input_tensor.dtype() == DataType::FLOAT32 || input_tensor.dtype() == DataType::BFLOAT16 ||
            input_tensor.dtype() == DataType::BFLOAT8_B,
        "Error");
    if (optional_input_tensors.size() == 1) {
        if (optional_input_tensors.at(0).has_value()) {
            auto& mask = optional_input_tensors.at(0).value();
            TT_FATAL(mask.storage_type() == StorageType::DEVICE, "Operands to softmax need to be on device!");
            TT_FATAL(input_tensor.device() == mask.device(), "Error");
            if (mask.is_sharded()) {  // sharded mask
                TT_FATAL(mask.layout() == Layout::TILE, "Error");
                TT_FATAL(mask.padded_shape() == input_tensor.padded_shape(), "Error");
            } else {
                if (mask.layout() == Layout::ROW_MAJOR) {
                    ttnn::Shape expected_shape(
                        {mask.padded_shape()[0], 1, input_tensor.padded_shape()[-1] / TILE_WIDTH, TILE_WIDTH});
                    TT_FATAL(mask.padded_shape() == expected_shape, "Error");
                }
                for (uint32_t i = 1; i < input_tensor.padded_shape().rank() - 2; i++) {
                    TT_FATAL(mask.padded_shape()[i] == 1, "Error");
                }
            }

            std::visit(
                [&](const auto& program_config) {
                    using ProgramConfigType = std::decay_t<decltype(program_config)>;
                    if constexpr (std::is_same_v<ProgramConfigType, SoftmaxDefaultProgramConfig>) {
                        TT_FATAL(input_tensor.padded_shape()[0] == mask.padded_shape()[0], "Error");
                        TT_FATAL(!this->is_scale_causal_mask_hw_dims_softmax, "Error");
                    } else if constexpr (std::is_same_v<ProgramConfigType, SoftmaxShardedMultiCoreProgramConfig>) {
                        const auto shape = input_tensor.padded_shape();
                        uint32_t M = input_tensor.physical_volume() / shape[-1];
                        uint32_t K = shape[-1];

                        TT_FATAL(M % TILE_HEIGHT == 0, "M must be divisible by tile height.");
                        TT_FATAL(K % TILE_WIDTH == 0, "K must be divisible by tile width.");
                        TT_FATAL(
                            program_config.block_w % program_config.subblock_w == 0,
                            "block_w must be divisible by subblock_w.");
                        TT_FATAL(
                            program_config.block_w * TILE_WIDTH == shape[3],
                            "shard width must equal to input tensor shape[3]!");
                        TT_FATAL(this->inplace, "Error");
                        if (!this->is_scale_causal_mask_hw_dims_softmax) {
                            // grid
                            auto num_cores_c = program_config.compute_with_storage_grid_size.x;
                            auto num_cores_r = program_config.compute_with_storage_grid_size.y;
                            // check dims
                            TT_FATAL(
                                M * K / ((program_config.block_w * program_config.block_h) * TILE_HW) ==
                                    num_cores_r * num_cores_c,
                                "number of shards must equal to number of cores. M = {}, K = {}, block_w = {}, block_h "
                                "= {}, num_cores = {}",
                                M,
                                K,
                                program_config.block_w,
                                program_config.block_h,
                                num_cores_r * num_cores_c);
                        } else {
                            TT_FATAL(this->is_causal_mask, "Error");
                            TT_FATAL(mask.layout() == Layout::TILE, "Error");
                            TT_FATAL(mask.is_sharded() == false, "Error");
                            TT_FATAL(input_tensor.layout() == Layout::TILE, "Error");
                            TT_FATAL(input_tensor.is_sharded(), "Error");
                            TT_FATAL(
                                input_tensor.shard_spec()->orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR,
                                "Error");
                            TT_FATAL(this->scale.has_value(), "Error");
                        }
                    }
                },
                this->program_config);
        } else {
            TT_FATAL(not this->scale.has_value(), "Error");
        }
    } else {
        TT_FATAL(not this->scale.has_value(), "Error");
        TT_FATAL(not this->is_scale_causal_mask_hw_dims_softmax, "Error");
    }
}

std::vector<TensorSpec> Softmax::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    auto& input_tensor = input_tensors.at(0);
    if (this->inplace) {
        return {input_tensor.tensor_spec()};
    }
    return {TensorSpec(
        input_tensor.logical_shape(),
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), output_mem_config))};
}

std::vector<Tensor> Softmax::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    if (this->inplace) {
        return {input_tensors.at(0)};
    }

    return {create_device_tensor(compute_output_specs(input_tensors)[0], input_tensors.at(0).device())};
}

tt::tt_metal::operation::ProgramWithCallbacks Softmax::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    const auto& mask = optional_input_tensors.at(0);
    // bool causal_mask = mask.has_value() ? mask.value().padded_shape()[-2] == mask.value().padded_shape()[-1]
    // : false;
    bool causal_mask = this->is_causal_mask;

    return std::visit(
        [&](const auto& program_config) -> tt::tt_metal::operation::ProgramWithCallbacks {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (std::is_same_v<ProgramConfigType, SoftmaxShardedMultiCoreProgramConfig>) {
                return scale_mask_softmax_sharded_multi_core(
                    input_tensor,
                    output_tensor,
                    mask,
                    this->scale,
                    causal_mask,
                    this->is_scale_causal_mask_hw_dims_softmax,
                    program_config.compute_with_storage_grid_size,
                    program_config.subblock_w,
                    program_config.block_h,
                    program_config.block_w,
                    this->compute_kernel_config,
                    this->numeric_stable);
            } else {
                return scale_mask_softmax_multi_core(
                    input_tensor,
                    output_tensor,
                    mask,
                    this->scale,
                    causal_mask,
                    this->compute_kernel_config,
                    this->numeric_stable);
            }
        },
        this->program_config);
}

tt::tt_metal::operation::Hash Softmax::compute_program_hash(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    return tt::tt_metal::operation::hash_operation<Softmax>(
        input_tensors.at(0).memory_config(),
        input_tensors.at(0).dtype(),
        optional_input_tensors.at(0).has_value() ? std::optional{optional_input_tensors.at(0).value().memory_config()}
                                                 : std::nullopt,
        optional_input_tensors.at(0).has_value() ? std::optional{optional_input_tensors.at(0).value().dtype()}
                                                 : std::nullopt,
        this->output_mem_config);
}

Tensor softmax_in_place(
    Tensor& input_tensor,
    const SoftmaxProgramConfig& program_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const bool numeric_stable) {
    return scale_mask_softmax_in_place(
        input_tensor, std::nullopt, std::nullopt, program_config, false, compute_kernel_config, numeric_stable);
}

Tensor scale_mask_softmax_in_place(
    Tensor& input_tensor,
    std::optional<float> scale,
    const std::optional<const Tensor>& mask,
    const SoftmaxProgramConfig& program_config,
    const bool is_causal_mask,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const bool numeric_stable) {
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor.device()->arch(), compute_kernel_config, MathFidelity::HiFi4, true, false, false);
    tt::tt_metal::operation::run(
        Softmax{
            .scale = scale,
            .inplace = true,
            .output_mem_config = input_tensor.memory_config(),
            .program_config = program_config,
            .is_causal_mask = is_causal_mask,
            .compute_kernel_config = kernel_config_val,
            .numeric_stable = numeric_stable},
        {input_tensor},
        {mask});

    return input_tensor;
}

Tensor scale_causal_mask_hw_dims_softmax_in_place(
    Tensor& input_tensor,
    std::optional<float> scale,
    const std::optional<const Tensor>& mask,
    const SoftmaxProgramConfig& program_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const bool numeric_stable) {
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor.device()->arch(), compute_kernel_config, MathFidelity::HiFi4, true, false, false);
    tt::tt_metal::operation::run(
        Softmax{
            .scale = scale,
            .inplace = true,
            .output_mem_config = input_tensor.memory_config(),
            .program_config = program_config,
            .is_causal_mask = true,
            .compute_kernel_config = kernel_config_val,
            .is_scale_causal_mask_hw_dims_softmax = true,
            .numeric_stable = numeric_stable},
        {input_tensor},
        {mask});
    return input_tensor;
}

Tensor softmax(
    const Tensor& input_tensor,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const bool numeric_stable) {
    return scale_mask_softmax(
        input_tensor, std::nullopt, std::nullopt, output_mem_config, false, compute_kernel_config, numeric_stable);
}

Tensor scale_mask_softmax(
    const Tensor& input_tensor,
    std::optional<float> scale,
    const std::optional<const Tensor>& mask,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const bool is_causal_mask,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const bool numeric_stable) {
    ttnn::Shape input_pad_shape =
        ttnn::operations::experimental::auto_format::AutoFormat::pad_to_tile_shape(input_tensor.padded_shape());
    ttnn::operations::experimental::auto_format::FormatParams input_format_params = {
        .pad_shape = input_pad_shape,
        .pad_value = -std::numeric_limits<float>::infinity(),
        .target_layout = tt::tt_metal::Layout::TILE};
    std::optional<ttnn::operations::experimental::auto_format::FormatParams> mask_format_params = std::nullopt;
    if (mask.has_value()) {
        TT_FATAL(input_tensor.padded_shape()[-1] == mask.value().padded_shape()[-1], "Error");
        TT_FATAL(input_tensor.padded_shape()[0] == mask.value().padded_shape()[0], "Error");
        TT_FATAL(mask.value().padded_shape()[-2] == 1 or mask.value().padded_shape()[-2] == TILE_HEIGHT, "Error");
        for (uint32_t i = 1; i < input_tensor.padded_shape().rank() - 2; i++) {
            TT_FATAL(mask.value().padded_shape()[i] == 1, "Error");
        }
        ttnn::Shape mask_pad_shape =
            ttnn::operations::experimental::auto_format::AutoFormat::pad_to_tile_shape(mask.value().padded_shape());
        mask_format_params = {
            .pad_shape = mask_pad_shape,
            .pad_value = -std::numeric_limits<float>::infinity(),
            .target_layout = tt::tt_metal::Layout::TILE};
    }
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor.device()->arch(), compute_kernel_config, MathFidelity::HiFi4, true, false, false);
    return tt::tt_metal::operation::run_with_autoformat(
               Softmax{
                   .scale = scale,
                   .inplace = false,
                   .output_mem_config = output_mem_config,
                   .is_causal_mask = is_causal_mask,
                   .compute_kernel_config = kernel_config_val,
                   .numeric_stable = numeric_stable},
               {input_tensor},
               {input_format_params},
               {tt::tt_metal::Layout::TILE},
               {mask},
               {mask_format_params})
        .at(0);
}

}  // namespace ttnn::operations::normalization
