// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_post_all_gather_op.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/math.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

#include <magic_enum.hpp>

#include <optional>

using uint32_t = std::uint32_t;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::normalization {

void LayerNormPostAllGather::validate(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_FATAL(input_tensors.size() == 2 and optional_input_tensors.size() <= 2, "Must have between 12 to 4 input tensors");
    auto& a = input_tensors.at(0);
    auto& stats = input_tensors.at(1);
    const auto& gamma = optional_input_tensors.at(0);
    const auto& beta = optional_input_tensors.at(1);

    for (const auto& tensor: input_tensors) {
        TT_FATAL(tensor.get_layout() == Layout::TILE, "Error");
        TT_FATAL(tensor.get_dtype() == DataType::BFLOAT16 || tensor.get_dtype() == DataType::BFLOAT8_B, "Error");
        TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "Operands to layernorm need to be on device!");
        TT_FATAL(tensor.buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
    }

    // stats has 2 or 1 tile columns per device if layernorm or rmsnorm
    TT_FATAL(stats.get_legacy_shape()[-1] % TILE_WIDTH == 0, "Error");
    TT_FATAL(stats.get_legacy_shape()[0] == a.get_legacy_shape()[0], "Error");
    TT_FATAL(stats.get_legacy_shape()[1] == a.get_legacy_shape()[1], "Error");
    TT_FATAL(stats.get_legacy_shape()[2] == a.get_legacy_shape()[2], "Error");
    // TODO: How to check if number of tile columns is correct? Would have to know # of devices and is_rmsnorm

    TT_FATAL(gamma.has_value(), "Error");
    const auto& gamma_tensor = gamma.value();

    TT_FATAL(gamma_tensor.get_layout() == Layout::ROW_MAJOR, "Error"); // Only support packed RM right now
    if (gamma_tensor.get_layout() == Layout::TILE) {
        TT_FATAL(a.get_legacy_shape()[-1] == gamma.value().get_legacy_shape()[-1], "{} != {}", a.get_legacy_shape()[-1], gamma.value().get_legacy_shape()[-1]);
        TT_FATAL(gamma.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
        TT_FATAL(a.device() == gamma.value().device(), "Error");
        TT_FATAL(gamma.value().get_legacy_shape()[-2] == TILE_HEIGHT, "Error");
    } else {
        TT_FATAL(gamma_tensor.get_layout() == Layout::ROW_MAJOR, "Error");
        TT_FATAL((gamma_tensor.get_legacy_shape()[-1] == TILE_WIDTH && gamma_tensor.volume() / TILE_WIDTH == a.get_legacy_shape()[-1] / TILE_WIDTH), "Error");
        TT_FATAL(gamma_tensor.buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
        TT_FATAL(a.device() == gamma_tensor.device(), "Error");
        TT_FATAL(gamma_tensor.get_dtype() == DataType::BFLOAT16, "Error");
    }
    const bool is_layernorm = this->norm_type == LayerNormDistributedType::LAYERNORM;
    const bool has_beta = beta.has_value();
    TT_FATAL(is_layernorm == has_beta, "Error"); // TODO: Is this a necessary check?

    if (beta.has_value()) {
        const auto& beta_tensor = beta.value();
        TT_FATAL(gamma_tensor.get_layout() == beta_tensor.get_layout(), "Gamma and beta must have the same layout!");
        TT_FATAL(beta_tensor.get_layout() == Layout::ROW_MAJOR, "Error");
        if (beta_tensor.get_layout() == Layout::TILE) {
            TT_FATAL(a.get_legacy_shape()[-1] == beta_tensor.get_legacy_shape()[-1], "Error");
            TT_FATAL(beta_tensor.buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
            TT_FATAL(a.device() == beta_tensor.device(), "Error");
            TT_FATAL(beta.value().get_legacy_shape()[-2] == TILE_HEIGHT, "Error");
        } else {
            TT_FATAL(beta_tensor.get_layout() == Layout::ROW_MAJOR, "Error");
            TT_FATAL((beta_tensor.get_legacy_shape()[-1] == TILE_WIDTH && beta_tensor.volume() / TILE_WIDTH == a.get_legacy_shape()[-1] / TILE_WIDTH), "Error");
            TT_FATAL(beta_tensor.buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
            TT_FATAL(a.device() == beta_tensor.device(), "Error");
            TT_FATAL(beta_tensor.get_dtype() == DataType::BFLOAT16, "Error");
        }
    }
}

std::vector<tt::tt_metal::LegacyShape> LayerNormPostAllGather::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.get_legacy_shape()};
}

std::vector<Tensor> LayerNormPostAllGather::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.get_dtype(), Layout::TILE, this->memory_config);
}

operation::ProgramWithCallbacks LayerNormPostAllGather::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor> &output_tensors
) const {
    const auto& a = input_tensors.at(0);
    const auto& stats = input_tensors.at(1);
    const auto& gamma = optional_input_tensors.at(0);
    const auto& beta = optional_input_tensors.at(1);
    auto& output_tensor = output_tensors.at(0);

    return layernorm_post_allgather_multi_core(
        a, stats, gamma, beta, output_tensor, this->norm_type, this->eps, this->compute_kernel_config
    );
}
}  // namespace ttnn::operations::normalization
