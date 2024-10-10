// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "groupnorm_op.hpp"

#include <optional>

#include "ttnn/operations/moreh/math.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"


using namespace tt::constants;

namespace ttnn::operations::normalization {

void GroupNorm::validate(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_FATAL(input_tensors.size() == 1 and optional_input_tensors.size() <= 3, "Must have between 1 to 4 input tensors");
    auto& a = input_tensors.at(0);
    const auto& gamma = optional_input_tensors.at(0);
    const auto& beta = optional_input_tensors.at(1);
    const auto& input_mask = optional_input_tensors.at(2);
    TT_FATAL(a.get_dtype() == DataType::BFLOAT16, "Error");
    TT_FATAL(a.storage_type() == StorageType::DEVICE, "Operands to layernorm need to be on device!");
    TT_FATAL(a.buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
    TT_FATAL(a.get_legacy_shape()[3] % this->num_groups == 0,  "channel must be divisible by num_groups!");
    TT_FATAL(a.get_legacy_shape()[1] == 1,  "input tensor shape[1] must be 1!");

    if (gamma.has_value()) {
        if (gamma.value().get_layout() == Layout::TILE) {
            TT_FATAL(a.get_legacy_shape()[3] == gamma.value().get_legacy_shape()[3], "{} != {}", a.get_legacy_shape()[3], gamma.value().get_legacy_shape()[3]);
            TT_FATAL(a.device() == gamma.value().device(), "Error");
            TT_FATAL(gamma.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
            TT_FATAL(gamma.value().get_legacy_shape()[2] == TILE_HEIGHT, "Error");
        } else {
            TT_FATAL(gamma.value().get_layout() == Layout::ROW_MAJOR, "Error");
            TT_FATAL((gamma.value().get_legacy_shape()[3] == TILE_WIDTH), "Error");
            TT_FATAL(a.device() == gamma.value().device(), "Error");
            TT_FATAL(gamma.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
            TT_FATAL(gamma.value().get_dtype() == DataType::BFLOAT16, "Error");
        }
        if (beta.has_value()) {
            TT_FATAL(gamma.value().get_layout() == beta.value().get_layout(), "Error");
        }
    }

    if (beta.has_value()) {
        if (beta.value().get_layout() == Layout::TILE) {
            TT_FATAL(a.get_legacy_shape()[3] == beta.value().get_legacy_shape()[3], "Error");
            TT_FATAL(a.device() == beta.value().device(), "Error");
            TT_FATAL(beta.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
            TT_FATAL(beta.value().get_legacy_shape()[2] == TILE_HEIGHT, "Error");
        } else {
            TT_FATAL(beta.value().get_layout() == Layout::ROW_MAJOR, "Error");
            TT_FATAL(beta.value().get_legacy_shape()[3] == TILE_WIDTH, "Error");
            TT_FATAL(a.device() == beta.value().device(), "Error");
            TT_FATAL(beta.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
            TT_FATAL(beta.value().get_dtype() == DataType::BFLOAT16, "Error");
        }
    }

    if (input_mask.has_value()) {
        TT_FATAL(input_mask.value().get_layout() == Layout::TILE, "Error");
        TT_FATAL(input_mask.value().get_legacy_shape()[1] == this->num_groups, "Error");
        TT_FATAL(input_mask.value().get_legacy_shape()[2] == TILE_HEIGHT, "Error");
        TT_FATAL(input_mask.value().get_legacy_shape()[3] % TILE_WIDTH == 0, "Error");
    }
}
std::vector<tt::tt_metal::LegacyShape> GroupNorm::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.get_legacy_shape()};
}
std::vector<Tensor> GroupNorm::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    if (this->program_config.inplace) {
        return {input_tensors.at(0)};
    } else {
        auto mem_config = this->output_mem_config;
        mem_config.shard_spec = input_tensor.shard_spec();
        return {create_device_tensor(this->compute_output_shapes(input_tensors).at(0), program_config.out_data_format, this->program_config.output_layout, input_tensor.device(), mem_config)};
    }
}
operation::ProgramWithCallbacks GroupNorm::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor> &output_tensors
) const {
    const auto& a = input_tensors.at(0);
    const auto& gamma = optional_input_tensors.at(0);
    const auto& beta = optional_input_tensors.at(1);
    const auto& input_mask = optional_input_tensors.at(2);
    auto& output_tensor = output_tensors.at(0);

    MathFidelity fidelity = this->program_config.math_fidelity;
    uint32_t num_cores_x = this->program_config.compute_with_storage_grid_size.x;
    uint32_t num_cores_y = this->program_config.compute_with_storage_grid_size.y;
    bool inplace = this->program_config.inplace;
    CoreCoord grid_size = CoreCoord(num_cores_x, num_cores_y);
    uint32_t batch = a.get_legacy_shape()[0];

    return groupnorm_multi_core_sharded(
                                a, gamma, beta, input_mask, output_tensor, this->eps,
                                this->num_groups, batch,
                                fidelity,
                                program_config.im_data_format,
                                program_config.compute_with_storage_grid_size,
                                inplace
                                );
}

}  // namespace ttnn::operations::normalization
