// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_op.hpp"
#include "moe_program_factory.hpp"

namespace ttnn::operations::reduction {

void MoeDeviceOperation::validate_with_output_tensors(const std::vector<Tensor> &input_tensors,
                                                      const std::vector<std::optional<Tensor>> &output_tensors) const {
    auto input_shape = input_tensors.at(0).get_legacy_shape();
    TT_FATAL(input_shape.rank() == 4, "Input shape must be 4D, got {}", input_shape.rank());
    TT_FATAL(this->k == 32, "K must be equal to 32, pad with -infinity if necessary to get 32, got {}", this->k);

    TT_FATAL(input_shape[-1] >= 64,
             "Input shape inner dim {} must be a multiple of 64, pad with -infinity if necessary",
             input_shape[-1]);
    TT_FATAL((input_shape[-1] & (input_shape[-1] - 1)) == 0,
             "Input shape inner dim {} must be a power of 2, pad with -infinity if necessary",
             input_shape[-1]);
    TT_FATAL((input_shape[0] * input_shape[1] * input_shape[2]) % 32 == 0,
             "Input height (combined input_shape[0-3]) {} must be a multiple of 32",
             input_shape[0] * input_shape[1] * input_shape[2]);

    TT_FATAL(this->output_mem_config.is_sharded() == false, "Sharded implementation not supported yet");
    TT_FATAL(input_tensors.at(0).get_layout() == Layout::TILE, "The input must be in tiled format");

    auto topk_shape = input_tensors.at(2).get_legacy_shape();
    auto expert_shape = input_tensors.at(1).get_legacy_shape();

    TT_FATAL(topk_shape[-1] == this->k, "Topk shape inner dim must be equal to k, got {}", topk_shape[-1]);
    TT_FATAL(expert_shape[-1] == input_shape[-1],
             "Expert shape inner dim must be equal to input_shape[-1], got {}",
             expert_shape[-1]);
    TT_FATAL(topk_shape[-2] == 32, "Topk shape inner dim must be equal to 32, got {}", topk_shape[-2]);
    TT_FATAL(expert_shape[-2] == 32, "Expert shape inner dim must be equal to 32, got {}", expert_shape[-2]);
}

std::vector<tt::tt_metal::LegacyShape> MoeDeviceOperation::compute_output_shapes(
    const std::vector<Tensor> &input_tensors) const {
    const auto &input_tensor = input_tensors.at(0);
    const auto input_shape = input_tensor.get_legacy_shape();
    return {ttnn::Shape(std::array<uint32_t, 4>{input_shape[0], input_shape[1], input_shape[2], 1},
                        std::array<uint32_t, 4>{input_shape[0], input_shape[1], input_shape[2], 32})
                .value};
}

std::vector<Tensor> MoeDeviceOperation::create_output_tensors(
    const std::vector<Tensor> &input_tensors,
    const std::vector<std::optional<Tensor>> &output_tensors) const {
    if (output_tensors.size() == 1) {
        if (output_tensors.at(0).has_value()) {
            return {output_tensors[0].value()};
        }
    }
    const auto &input_tensor = input_tensors.at(0);
    const auto shapes = compute_output_shapes(input_tensors);
    auto out_tensor = create_device_tensor(
        shapes[0], input_tensor.get_dtype(), Layout::TILE, input_tensor.device(), this->output_mem_config);
    return {out_tensor};
}

operation::ProgramWithCallbacks MoeDeviceOperation::create_program(const std::vector<Tensor> &input_tensors,
                                                                   std::vector<Tensor> &output_tensors) const {
    return detail::moe_single_core_interleaved(
        input_tensors.at(0), input_tensors.at(1), input_tensors.at(2), this->k, output_tensors.at(0));
}

}  // namespace ttnn::operations::reduction
