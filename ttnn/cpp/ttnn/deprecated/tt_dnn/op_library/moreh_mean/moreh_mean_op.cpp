// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/op_library/moreh_mean/moreh_mean_op.hpp"

#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"

namespace tt {
using namespace constants;
namespace operations {
namespace primary {

////////////////////////////////////////////////////////////////////////////
//                         MorehMean
////////////////////////////////////////////////////////////////////////////
void MorehMean::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    TT_FATAL((dim >= 0 && dim <= 7), "dim should be 0 - 7");
    TT_FATAL(input_tensors.size() == 1, "Error");
    TT_FATAL(this->divisor.has_value() == false, "divisor not supported yet.");

    const auto& input = input_tensors.at(0);
    auto& output = output_tensors.at(0);

    check_tensor(input, "moreh_mean", "input", {DataType::BFLOAT16});
    check_tensor(output, "moreh_mean", "output", {DataType::BFLOAT16});

    validate_input_with_dim(input, this->dim);

    if (output.has_value()) {
        validate_output_with_keepdim(input, output.value(), this->dim, this->keepdim);
    }
}

std::vector<Shape> MorehMean::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    auto input_shape = input_tensors[0].get_legacy_shape();
    auto output_shape = input_shape;
    auto input_rank = input_shape.rank();

    if (this->keepdim) {
        auto padding = output_shape.padding();
        if (dim + 1 == input_rank) {
            output_shape[dim] = TILE_WIDTH;
            padding[dim] = Padding::PadDimension{0, 31};
        } else if (dim + 2 == input_rank) {
            output_shape[dim] = TILE_HEIGHT;
            padding[dim] = Padding::PadDimension{0, 31};
        } else {
            output_shape[dim] = 1;
        }

        return {Shape(output_shape, padding)};
    }

    std::vector<uint32_t> shape;
    std::vector<Padding::PadDimension> pad_dimensions;
    const bool is_tile_dim = (this->dim == input_rank - 1 || this->dim == input_rank - 2);
    const std::size_t output_rank = (is_tile_dim) ? (input_rank) : (input_rank - 1);
    auto input_padding = input_shape.padding();

    // e.g. (2, 64, 64) with dim 1 to be (2, 1[32], 64)
    // e.g. (2, 64, 64) with dim 0 to be (64, 64)
    for (int i = 0; i < input_rank; ++i) {
        bool is_reduced_dim = (i == this->dim);
        if (is_reduced_dim && !is_tile_dim)
            continue;

        shape.push_back((is_reduced_dim && is_tile_dim) ? (TILE_HEIGHT) : (input_shape[i]));
        pad_dimensions.push_back((is_reduced_dim && is_tile_dim) ? (Padding::PadDimension{0, 31}) : (input_padding[i]));
    }

    auto padding = Padding(pad_dimensions, input_padding.pad_value());
    return {Shape(shape, padding)};
}

std::vector<Tensor> MorehMean::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (output_tensors[0].has_value()) {
        return {output_tensors[0].value()};
    }

    return operation::generic_create_output_tensors(
        *this, input_tensors, input_tensors[0].get_dtype(), Layout::TILE, this->memory_config);
}

operation::ProgramWithCallbacks MorehMean::create_program(
    const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) const {
    auto& input = inputs[0];
    auto& output = outputs[0];

    auto rank = input.get_legacy_shape().rank();

    if (dim + 1 == rank) {
        return moreh_mean_w(input, output, this->core_range, this->compute_kernel_config);
    } else if (dim + 2 == rank) {
        return moreh_mean_h(input, output, this->core_range, this->compute_kernel_config);
    }

    return moreh_mean_nc(input, output, dim, this->core_range, this->compute_kernel_config);
}

// output creation inside
Tensor moreh_mean_(
    const Tensor& input,
    const int64_t& dim,
    const bool keepdim,
    const std::optional<uint32_t> divisor,
    const std::optional<const Tensor> output,
    const std::optional<MemoryConfig> memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input}))};

    auto device = input.device();
    auto kernel_config_val =
        init_device_compute_kernel_config(device->arch(), compute_kernel_config, MathFidelity::HiFi4);

    auto grid_coord = device->compute_with_storage_grid_size();
    const CoreRange all_cores({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});

    operation::launch_op(
        [dim, keepdim, divisor, memory_config, all_cores, kernel_config_val](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            return operation::run(
                MorehMean{
                    .dim = dim,
                    .keepdim = keepdim,
                    .divisor = divisor,
                    .memory_config = memory_config.value_or(input_tensors.at(0).memory_config()),
                    .core_range = all_cores,
                    .compute_kernel_config = kernel_config_val},
                input_tensors,
                optional_input_tensors,
                optional_output_tensors);
        },
        {input},
        output_tensors,
        {},
        {output});

    if (output.has_value()) {
        return output.value();
    }

    return output_tensors[0];
}

std::optional<Tensor> moreh_mean(
    const Tensor& input,
    std::optional<std::variant<int64_t, std::vector<int64_t>>> dim,
    const bool keepdim,
    const std::optional<uint32_t> divisor,
    const std::optional<const Tensor> output,
    const std::optional<MemoryConfig> memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    auto rank = input.get_legacy_shape().rank();
    std::vector<int64_t> dims = get_dim(dim, rank);
    std::sort(dims.begin(), dims.end());

    auto temp_input = input;
    for (uint32_t i = dims.size() - 1; i > 0; i--) {
        log_debug(LogTest, "{}:{} dim {}", __func__, __LINE__, dims[i]);
        auto temp_output =
            moreh_mean_(temp_input, dims[i], keepdim, divisor, std::nullopt, memory_config, compute_kernel_config);
        temp_input = temp_output;
    }
    log_debug(LogTest, "{}:{} dim {}", __func__, __LINE__, dims.front());
    moreh_mean_(temp_input, dims.front(), keepdim, divisor, output, memory_config, compute_kernel_config);

    return output;
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
