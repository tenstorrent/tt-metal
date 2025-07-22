// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_post_all_gather_op.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/math.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>

#include <optional>

using uint32_t = std::uint32_t;
using namespace tt::constants;

namespace ttnn::operations::normalization {

void LayerNormPostAllGather::validate(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_FATAL(
        input_tensors.size() == 2 and optional_input_tensors.size() <= 2, "Must have between 12 to 4 input tensors");
    auto& a = input_tensors.at(0);
    auto& stats = input_tensors.at(1);
    const auto& gamma = optional_input_tensors.at(0);
    const auto& beta = optional_input_tensors.at(1);

    for (const auto& tensor : input_tensors) {
        TT_FATAL(tensor.layout() == Layout::TILE, "Error");
        TT_FATAL(tensor.dtype() == DataType::BFLOAT16 || tensor.dtype() == DataType::BFLOAT8_B, "Error");
        TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "Operands to layernorm need to be on device!");
        TT_FATAL(tensor.buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
    }

    // stats has 2 or 1 tile columns per device if layernorm or rmsnorm
    TT_FATAL(stats.padded_shape()[-1] % TILE_WIDTH == 0, "Error");
    TT_FATAL(stats.padded_shape()[0] == a.padded_shape()[0], "Error");
    TT_FATAL(stats.padded_shape()[1] == a.padded_shape()[1], "Error");
    TT_FATAL(stats.padded_shape()[2] == a.padded_shape()[2], "Error");
    // TODO: How to check if number of tile columns is correct? Would have to know # of devices and is_rmsnorm

    if (gamma.has_value()) {
        const auto& gamma_tensor = gamma.value();

        TT_FATAL(gamma_tensor.layout() == Layout::ROW_MAJOR, "Error");  // Only support packed RM right now
        if (gamma_tensor.layout() == Layout::TILE) {
            TT_FATAL(
                a.padded_shape()[-1] == gamma.value().padded_shape()[-1],
                "{} != {}",
                a.padded_shape()[-1],
                gamma.value().padded_shape()[-1]);
            TT_FATAL(
                gamma.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
            TT_FATAL(a.device() == gamma.value().device(), "Error");
            TT_FATAL(gamma.value().padded_shape()[-2] == TILE_HEIGHT, "Error");
        } else {
            TT_FATAL(gamma_tensor.layout() == Layout::ROW_MAJOR, "Error");
            TT_FATAL(
                (gamma_tensor.padded_shape()[-1] == TILE_WIDTH &&
                 gamma_tensor.physical_volume() / TILE_WIDTH == a.padded_shape()[-1] / TILE_WIDTH),
                "Error");
            TT_FATAL(
                gamma_tensor.buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
            TT_FATAL(a.device() == gamma_tensor.device(), "Error");
            TT_FATAL(gamma_tensor.dtype() == DataType::BFLOAT16, "Error");
        }
        const bool is_layernorm = this->norm_type == LayerNormDistributedType::LAYERNORM;
        const bool has_beta = beta.has_value();
        TT_FATAL(is_layernorm == has_beta, "Error");  // TODO: Is this a necessary check?

        if (beta.has_value()) {
            const auto& beta_tensor = beta.value();
            TT_FATAL(gamma_tensor.layout() == beta_tensor.layout(), "Gamma and beta must have the same layout!");
            TT_FATAL(beta_tensor.layout() == Layout::ROW_MAJOR, "Error");
            if (beta_tensor.layout() == Layout::TILE) {
                TT_FATAL(a.padded_shape()[-1] == beta_tensor.padded_shape()[-1], "Error");
                TT_FATAL(
                    beta_tensor.buffer() != nullptr,
                    "Operands to layernorm need to be allocated in buffers on device!");
                TT_FATAL(a.device() == beta_tensor.device(), "Error");
                TT_FATAL(beta.value().padded_shape()[-2] == TILE_HEIGHT, "Error");
            } else {
                TT_FATAL(beta_tensor.layout() == Layout::ROW_MAJOR, "Error");
                TT_FATAL(
                    (beta_tensor.padded_shape()[-1] == TILE_WIDTH &&
                     beta_tensor.physical_volume() / TILE_WIDTH == a.padded_shape()[-1] / TILE_WIDTH),
                    "Error");
                TT_FATAL(
                    beta_tensor.buffer() != nullptr,
                    "Operands to layernorm need to be allocated in buffers on device!");
                TT_FATAL(a.device() == beta_tensor.device(), "Error");
                TT_FATAL(beta_tensor.dtype() == DataType::BFLOAT16, "Error");
            }
        }
    }
}

std::vector<TensorSpec> LayerNormPostAllGather::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    auto& input_tensor = input_tensors.at(0);
    return {TensorSpec(
        input_tensor.logical_shape(),
        tt::tt_metal::TensorLayout(
            this->dtype.value_or(input_tensor.dtype()), tt::tt_metal::PageConfig(Layout::TILE), memory_config))};
}

tt::tt_metal::operation::ProgramWithCallbacks LayerNormPostAllGather::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    const auto& a = input_tensors.at(0);
    const auto& stats = input_tensors.at(1);
    const auto& gamma = optional_input_tensors.at(0);
    const auto& beta = optional_input_tensors.at(1);
    auto& output_tensor = output_tensors.at(0);

    return layernorm_post_allgather_multi_core(
        a, stats, gamma, beta, output_tensor, this->norm_type, this->eps, this->compute_kernel_config);
}
}  // namespace ttnn::operations::normalization
