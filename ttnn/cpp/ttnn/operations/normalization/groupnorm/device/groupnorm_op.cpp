// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "groupnorm_op.hpp"
#include "groupnorm_types.hpp"

#include <optional>

#include "ttnn/operations/math.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::normalization {

void GroupNorm::validate(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_FATAL(
        input_tensors.size() == 1 and optional_input_tensors.size() <= 4, "Must have between 1 to 5 input tensors");
    auto& a = input_tensors.at(0);
    const auto& gamma = optional_input_tensors.at(0);
    const auto& beta = optional_input_tensors.at(1);
    const auto& input_mask = optional_input_tensors.at(2);
    const auto& negative_mask = optional_input_tensors.at(3);
    TT_FATAL(a.dtype() == DataType::BFLOAT16, "Error");
    TT_FATAL(a.storage_type() == StorageType::DEVICE, "Operands to groupnorm need to be on device!");
    TT_FATAL(a.buffer() != nullptr, "Operands to groupnorm need to be allocated in buffers on device!");
    TT_FATAL(a.padded_shape()[3] % this->num_groups == 0, "channel must be divisible by num_groups!");
    TT_FATAL(a.padded_shape()[1] == 1, "input tensor shape[1] must be 1!");

    if (gamma.has_value()) {
        if (gamma.value().layout() == Layout::TILE) {
            TT_FATAL(
                a.padded_shape()[3] == gamma.value().padded_shape()[3],
                "{} != {}",
                a.padded_shape()[3],
                gamma.value().padded_shape()[3]);
            TT_FATAL(a.device() == gamma.value().device(), "Error");
            TT_FATAL(
                gamma.value().buffer() != nullptr, "Operands to groupnorm need to be allocated in buffers on device!");
            TT_FATAL(gamma.value().padded_shape()[2] == TILE_HEIGHT, "Error");
        } else {
            TT_FATAL(gamma.value().layout() == Layout::ROW_MAJOR, "Error");
            TT_FATAL((gamma.value().padded_shape()[3] == TILE_WIDTH), "Error");
            TT_FATAL(a.device() == gamma.value().device(), "Error");
            TT_FATAL(
                gamma.value().buffer() != nullptr, "Operands to groupnorm need to be allocated in buffers on device!");
            TT_FATAL(gamma.value().dtype() == DataType::BFLOAT16, "Error");
        }
        if (beta.has_value()) {
            TT_FATAL(gamma.value().layout() == beta.value().layout(), "Error");
        }
    }

    if (beta.has_value()) {
        if (beta.value().layout() == Layout::TILE) {
            TT_FATAL(a.padded_shape()[3] == beta.value().padded_shape()[3], "Error");
            TT_FATAL(a.device() == beta.value().device(), "Error");
            TT_FATAL(
                beta.value().buffer() != nullptr, "Operands to groupnorm need to be allocated in buffers on device!");
            TT_FATAL(beta.value().padded_shape()[2] == TILE_HEIGHT, "Error");
        } else {
            TT_FATAL(beta.value().layout() == Layout::ROW_MAJOR, "Error");
            TT_FATAL(beta.value().padded_shape()[3] == TILE_WIDTH, "Error");
            TT_FATAL(a.device() == beta.value().device(), "Error");
            TT_FATAL(
                beta.value().buffer() != nullptr, "Operands to groupnorm need to be allocated in buffers on device!");
            TT_FATAL(beta.value().dtype() == DataType::BFLOAT16, "Error");
        }
    }

    if (input_mask.has_value()) {
        TT_FATAL(input_mask.value().layout() == Layout::TILE, "Error");
        TT_FATAL(input_mask.value().padded_shape()[1] == this->num_groups, "Error");
        TT_FATAL(input_mask.value().padded_shape()[2] == TILE_HEIGHT, "Error");
        TT_FATAL(input_mask.value().padded_shape()[3] % TILE_WIDTH == 0, "Error");
    }

    // Negative mask tensor is used to reduce the number of CB's used in the sharded version of the kernel by
    // overlapping the CB's used for tilized input and output. (The kernel is in fact row major variant, but is
    // internally tilizing RM into tilized inputs) Valid only if sharded program is used, and input and output tensors
    // are in row major layout.
    if (negative_mask.has_value()) {
        TT_FATAL(
            negative_mask.value().layout() == Layout::TILE,
            "Negative musk must be in TILE layout, but layout is {}",
            negative_mask.value().layout());
        TT_FATAL(
            negative_mask.value().padded_shape()[1] == this->num_groups,
            "Negative mask padded shape[1] must be equal to num_groups, but is {} and num_groups is {}",
            negative_mask.value().padded_shape()[1],
            this->num_groups);
        TT_FATAL(
            negative_mask.value().padded_shape()[2] == TILE_HEIGHT,
            "Negative mask padded shape[2] must be equal to TILE_HEIGHT, but is {} and TILE_HEIGHT is {}",
            negative_mask.value().padded_shape()[2],
            TILE_HEIGHT);
        TT_FATAL(
            negative_mask.value().padded_shape()[3] % TILE_WIDTH == 0,
            "Negative mask padded shape[3] must be divisible by TILE_WIDTH, but is {} and TILE_WIDTH is {}",
            negative_mask.value().padded_shape()[3],
            TILE_WIDTH);
        TT_FATAL(a.is_sharded(), "Negative mask support is only available for sharded input tensors.");
        TT_FATAL(
            a.layout() == Layout::ROW_MAJOR,
            "If using negative mask, input tensor must be in ROW_MAJOR layout, but layout is {}",
            a.layout());
        Layout output_layout =
            std::visit([](const auto& config) -> Layout { return config.output_layout; }, this->program_config);
        TT_FATAL(
            output_layout == Layout::ROW_MAJOR,
            "If using negative mask, output tensor must be in ROW_MAJOR layout, but layout is {}",
            output_layout);
    }
}
std::vector<TensorSpec> GroupNorm::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);

    return std::visit(
        [&](const auto& program_config) -> std::vector<TensorSpec> {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if (program_config.inplace) {
                if constexpr (std::is_same_v<ProgramConfigType, GroupNormShardedMultiCoreProgramConfig>) {
                    return {input_tensor.tensor_spec()};
                } else {
                    TT_THROW("inplace groupnorm not supported for unsharded tensors");
                }
            }

            auto mem_config = this->output_mem_config;
            return {TensorSpec(
                input_tensor.logical_shape(),
                TensorLayout(program_config.out_data_format, PageConfig(program_config.output_layout), mem_config))};
        },
        this->program_config);
}
std::vector<Tensor> GroupNorm::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);

    return std::visit(
        [&](const auto& program_config) -> std::vector<Tensor> {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if (program_config.inplace) {
                if constexpr (std::is_same_v<ProgramConfigType, GroupNormShardedMultiCoreProgramConfig>) {
                    return {input_tensor};
                } else {
                    TT_THROW("inplace groupnorm not supported for unsharded tensors");
                }
            }
            return {create_device_tensor(this->compute_output_specs(input_tensors).at(0), input_tensor.device())};
        },
        this->program_config);
}

operation::ProgramWithCallbacks GroupNorm::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    const auto& a = input_tensors.at(0);
    const auto& gamma = optional_input_tensors.at(0);
    const auto& beta = optional_input_tensors.at(1);
    const auto& input_mask = optional_input_tensors.at(2);
    const auto& negative_mask = optional_input_tensors.at(3);
    auto& output_tensor = output_tensors.at(0);

    return std::visit(
        [&](const auto& program_config) -> operation::ProgramWithCallbacks {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (std::is_same_v<ProgramConfigType, GroupNormShardedMultiCoreProgramConfig>) {
                uint32_t num_cores_x = program_config.compute_with_storage_grid_size.x;
                uint32_t num_cores_y = program_config.compute_with_storage_grid_size.y;
                bool inplace = program_config.inplace;
                CoreCoord grid_size = CoreCoord(num_cores_x, num_cores_y);
                uint32_t batch = a.padded_shape()[0];

                return groupnorm_multi_core_sharded(
                    a,
                    gamma,
                    beta,
                    input_mask,
                    negative_mask,
                    output_tensor,
                    this->eps,
                    this->num_groups,
                    batch,
                    program_config.im_data_format,
                    program_config.compute_with_storage_grid_size,
                    inplace,
                    this->compute_kernel_config);
            } else {
                uint32_t num_cores_x = program_config.compute_with_storage_grid_size.x;
                uint32_t num_cores_y = program_config.compute_with_storage_grid_size.y;
                bool inplace = program_config.inplace;
                uint32_t num_out_blocks = program_config.num_out_blocks;
                CoreCoord grid_size = CoreCoord(num_cores_x, num_cores_y);
                uint32_t batch = a.padded_shape()[0];

                return groupnorm_multi_core(
                    a,
                    gamma,
                    beta,
                    input_mask,
                    output_tensor,
                    this->eps,
                    this->num_groups,
                    batch,
                    program_config.im_data_format,
                    program_config.compute_with_storage_grid_size,
                    inplace,
                    num_out_blocks,
                    this->compute_kernel_config);
            }
        },
        this->program_config);
}

}  // namespace ttnn::operations::normalization
