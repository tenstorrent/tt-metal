// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::conv {
namespace conv2d {

// TODO: Accept parallelization
enum class ConvOpParallelizationStrategy {
    MULTI_CORE, MULTI_CORE_REUSE, MULTI_CORE_REUSE_MCAST, SINGLE_CORE
};

struct Conv {
     // additional parameters
    const std::vector<int> conv_params;
    const uint32_t act_block_h_ntiles, act_block_w_ntiles, weight_block_w_ntiles, out_subblock_h_ntiles, out_subblock_w_ntiles, output_channels;
    bool use_address_map, use_fast_reader, untilize_out, has_bias, fuse_relu;
    MathFidelity math_fidelity;
    Conv(uint32_t act_bh, uint32_t act_bw, uint32_t weight_bw, uint32_t out_sh, uint32_t out_sw, const std::vector<int>&c_params, uint32_t output_channels, bool address_map, bool fast_reader, bool untile_out, bool has_bias, bool fuse_relu, MathFidelity mfidelity)
        : act_block_h_ntiles(act_bh),
          act_block_w_ntiles(act_bw),
          weight_block_w_ntiles(weight_bw),
          out_subblock_h_ntiles(out_sh),
          out_subblock_w_ntiles(out_sw),
          output_channels(output_channels),
          conv_params(c_params),
          use_address_map(address_map),
          use_fast_reader(fast_reader),
          untilize_out(untile_out),
          has_bias(has_bias),
          fuse_relu(fuse_relu),
          math_fidelity(mfidelity) {}

    void validate(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
    std::vector<tt::tt_metal::LegacyShape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, std::vector<Tensor> &output_tensors) const;

    static constexpr auto attribute_names = std::make_tuple(
        "conv_params",
        "act_block_h_ntiles",
        "act_block_w_ntiles",
        "weight_block_w_ntiles",
        "out_subblock_h_ntiles",
        "out_subblock_w_ntiles",
        "output_channels",
        "use_address_map",
        "use_fast_reader",
        "untilize_out",
        "has_bias",
        "fuse_relu",
        "math_fidelity");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->conv_params),
            std::cref(this->act_block_h_ntiles),
            std::cref(this->act_block_w_ntiles),
            std::cref(this->weight_block_w_ntiles),
            std::cref(this->out_subblock_h_ntiles),
            std::cref(this->out_subblock_w_ntiles),
            std::cref(this->output_channels),
            std::cref(this->use_address_map),
            std::cref(this->use_fast_reader),
            std::cref(this->untilize_out),
            std::cref(this->has_bias),
            std::cref(this->fuse_relu),
            std::cref(this->math_fidelity));
    }
};

Tensor conv(const Tensor& a, const Tensor &b, std::optional<const Tensor> bias, const vector<int> conv_params, uint32_t act_block_h_ntiles, uint32_t act_block_w_ntiles, uint32_t weight_block_w_ntiles,
             uint32_t out_subblock_h_ntiles, uint32_t out_subblock_w_ntiles, uint32_t output_channels, bool has_bias);

Tensor conv_with_fast_reader(const Tensor& a, const Tensor &b, std::optional<const Tensor> bias, const vector<int> conv_params, uint32_t act_block_h_ntiles, uint32_t act_block_w_ntiles, uint32_t weight_block_w_ntiles,
             uint32_t out_subblock_h_ntiles, uint32_t out_subblock_w_ntiles, uint32_t output_channels, bool untilize_out, bool has_bias, bool fuse_relu, MathFidelity math_fidelity = MathFidelity::HiFi4);

operation::ProgramWithCallbacks conv_single_core(const Tensor& A, const Tensor& B, std::optional<const Tensor> bias, vector<int> conv_params, uint32_t act_block_h_ntiles, uint32_t act_block_w_ntiles, uint32_t weight_block_w_ntiles,
             uint32_t out_subblock_h_ntiles, uint32_t out_subblock_w_ntiles, uint32_t output_channels, bool has_bias, MathFidelity math_fidelity, Tensor& output); // Tilizes a, untilizes b

Tensor conv_with_address_map(const Tensor& a, const Tensor &b, std::optional<const Tensor> bias, const vector<int> conv_params, uint32_t act_block_h_ntiles, uint32_t act_block_w_ntiles, uint32_t weight_block_w_ntiles,
             uint32_t out_subblock_h_ntiles, uint32_t out_subblock_w_ntiles, uint32_t output_channels);
operation::ProgramWithCallbacks conv_with_address_map_single_core(const Tensor& A, const Tensor& B, vector<int> conv_params, uint32_t act_block_h_ntiles, uint32_t act_block_w_ntiles, uint32_t weight_block_w_ntiles,
             uint32_t out_subblock_h_ntiles, uint32_t out_subblock_w_ntiles, uint32_t output_channels, Tensor& output); // Tilizes a, untilizes b


}  // namespace tt_metal

}  // namespace tt

// TODO: Merge with optimized_conv_op_utils?
namespace conv_op_utils {
using namespace tt;
using namespace tt::tt_metal;

pair<uint32_t, uint32_t> compute_conv_output_face_shape(uint32_t conv_activation_h, uint32_t conv_activation_w, uint32_t filter_h, uint32_t filter_w, uint32_t stride_h, uint32_t stride_w, uint32_t pad_h, uint32_t pad_w);
pair<vector<uint32_t>, vector<uint32_t>> compute_conv_activation_as_mm_shape(tt::tt_metal::LegacyShape conv_activation_shape, vector<int> conv_params, uint32_t act_block_h_ntiles, uint32_t act_block_w_ntiles, bool use_fast_reader);

}
