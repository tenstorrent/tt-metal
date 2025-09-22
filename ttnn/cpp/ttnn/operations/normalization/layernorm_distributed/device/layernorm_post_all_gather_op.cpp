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
        TT_FATAL(tensor.layout() == Layout::TILE, "Input tensor must have TILE layout, got: {}", tensor.layout());
        TT_FATAL(
            tensor.dtype() == DataType::BFLOAT16 || tensor.dtype() == DataType::BFLOAT8_B,
            "Input tensor must be BFLOAT16 or BFLOAT8_B, got: {}",
            tensor.dtype());
        TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "Operands to layernorm need to be on device!");
        TT_FATAL(tensor.buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
    }

    // stats has 2 or 1 tile columns per device if layernorm or rmsnorm
    TT_FATAL(
        stats.padded_shape()[-1] % TILE_WIDTH == 0,
        "Stats inner dimension must be divisible by TILE_WIDTH (32), got: {}",
        stats.padded_shape()[-1]);
    TT_FATAL(
        stats.padded_shape()[0] == a.padded_shape()[0],
        "Stats and input batch sizes must match, got stats: {} vs input: {}",
        stats.padded_shape()[0],
        a.padded_shape()[0]);
    TT_FATAL(
        stats.padded_shape()[1] == a.padded_shape()[1],
        "Stats and input dim1 must match, got stats: {} vs input: {}",
        stats.padded_shape()[1],
        a.padded_shape()[1]);
    TT_FATAL(
        stats.padded_shape()[2] == a.padded_shape()[2],
        "Stats and input dim2 must match, got stats: {} vs input: {}",
        stats.padded_shape()[2],
        a.padded_shape()[2]);
    // TODO: How to check if number of tile columns is correct? Would have to know # of devices and is_rmsnorm

    if (gamma.has_value()) {
        const auto& gamma_tensor = gamma.value();

        TT_FATAL(
            gamma_tensor.layout() == Layout::ROW_MAJOR,
            "Gamma tensor must have ROW_MAJOR layout (only packed RM supported), got: {}",
            gamma_tensor.layout());
        if (gamma_tensor.layout() == Layout::TILE) {
            TT_FATAL(
                a.padded_shape()[-1] == gamma.value().padded_shape()[-1],
                "{} != {}",
                a.padded_shape()[-1],
                gamma.value().padded_shape()[-1]);
            TT_FATAL(
                gamma.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
            TT_FATAL(a.device() == gamma.value().device(), "Input and gamma tensors must be on same device");
            TT_FATAL(
                gamma.value().padded_shape()[-2] == TILE_HEIGHT,
                "Gamma tensor height must be TILE_HEIGHT (32), got: {}",
                gamma.value().padded_shape()[-2]);
        } else {
            TT_FATAL(
                gamma_tensor.layout() == Layout::ROW_MAJOR,
                "Gamma tensor must have ROW_MAJOR layout, got: {}",
                gamma_tensor.layout());
            TT_FATAL(
                (gamma_tensor.padded_shape()[-1] == TILE_WIDTH &&
                 gamma_tensor.physical_volume() / TILE_WIDTH == a.padded_shape()[-1] / TILE_WIDTH),
                "Gamma tensor dimensions must align with input tensor. Got gamma padded shape: {}, physical volume: "
                "{}, input padded shape: {}, TILE_WIDTH: {}",
                gamma_tensor.padded_shape()[-1],
                gamma_tensor.physical_volume(),
                a.padded_shape()[-1],
                TILE_WIDTH);
            TT_FATAL(
                gamma_tensor.buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
            TT_FATAL(a.device() == gamma_tensor.device(), "Input and gamma tensors must be on same device");
            TT_FATAL(
                gamma_tensor.dtype() == DataType::BFLOAT16,
                "Gamma tensor must be BFLOAT16, got: {}",
                gamma_tensor.dtype());
        }
        const bool is_layernorm = this->norm_type == LayerNormDistributedType::LAYERNORM;
        const bool has_beta = beta.has_value();
        TT_FATAL(is_layernorm == has_beta, "Beta tensor must be present if and only if using layernorm (vs rmsnorm)");

        if (beta.has_value()) {
            const auto& beta_tensor = beta.value();
            TT_FATAL(
                gamma_tensor.layout() == beta_tensor.layout(),
                "Gamma and beta must have the same layout, got gamma: {} vs beta: {}",
                gamma_tensor.layout(),
                beta_tensor.layout());
            TT_FATAL(
                beta_tensor.layout() == Layout::ROW_MAJOR,
                "Beta tensor must have ROW_MAJOR layout, got: {}",
                beta_tensor.layout());
            if (beta_tensor.layout() == Layout::TILE) {
                TT_FATAL(
                    a.padded_shape()[-1] == beta_tensor.padded_shape()[-1],
                    "Input and beta inner dimensions must match, got input: {} vs beta: {}",
                    a.padded_shape()[-1],
                    beta_tensor.padded_shape()[-1]);
                TT_FATAL(
                    beta_tensor.buffer() != nullptr,
                    "Operands to layernorm need to be allocated in buffers on device!");
                TT_FATAL(a.device() == beta_tensor.device(), "Input and beta tensors must be on same device");
                TT_FATAL(
                    beta.value().padded_shape()[-2] == TILE_HEIGHT,
                    "Beta tensor height must be TILE_HEIGHT (32), got: {}",
                    beta.value().padded_shape()[-2]);
            } else {
                TT_FATAL(
                    beta_tensor.layout() == Layout::ROW_MAJOR,
                    "Beta tensor must have ROW_MAJOR layout, got: {}",
                    beta_tensor.layout());
                TT_FATAL(
                    (beta_tensor.padded_shape()[-1] == TILE_WIDTH &&
                     beta_tensor.physical_volume() / TILE_WIDTH == a.padded_shape()[-1] / TILE_WIDTH),
                    "Beta tensor dimensions must align with input tensor. Got beta padded shape: {}, physical volume: "
                    "{}, input padded shape: {}, TILE_WIDTH: {}",
                    beta_tensor.padded_shape()[-1],
                    beta_tensor.physical_volume(),
                    a.padded_shape()[-1],
                    TILE_WIDTH);
                TT_FATAL(
                    beta_tensor.buffer() != nullptr,
                    "Operands to layernorm need to be allocated in buffers on device!");
                TT_FATAL(a.device() == beta_tensor.device(), "Input and beta tensors must be on same device");
                TT_FATAL(
                    beta_tensor.dtype() == DataType::BFLOAT16,
                    "Beta tensor must be BFLOAT16, got: {}",
                    beta_tensor.dtype());
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
        a,
        stats,
        gamma,
        beta,
        output_tensor,
        this->norm_type,
        this->eps,
        this->compute_kernel_config,
        this->use_2d_core_grid);
}
}  // namespace ttnn::operations::normalization
