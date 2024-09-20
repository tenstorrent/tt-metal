// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/op_library/moreh_norm/moreh_norm_op.hpp"

#include <algorithm>
#include <functional>
#include <numeric>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_metal/common/work_split.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {

namespace operations {

namespace primary {

namespace {

inline void validate_input_tensor_with_dim(const Tensor& input, const int64_t &dim) {
    auto input_shape = input.get_legacy_shape();
    auto input_shape_wo_padding = input.get_legacy_shape().without_padding();
    const auto input_rank = input_shape.rank();
    log_debug(LogOp, "{}:{} input_rank {}", __func__, __LINE__, input_rank);
    TT_FATAL(
        (dim >= 0 && dim <= tt::tt_metal::MAX_NUM_DIMENSIONS),
        "dim must be between 0 and {}.",
        tt::tt_metal::MAX_NUM_DIMENSIONS);
    TT_FATAL((dim < input_rank), "dim must be smaller than input tensor rank {}.", input_rank);
}


inline void validate_output_tensor_with_keepdim(const Tensor& input, const Tensor& output, const int64_t &dim, const bool &keepdim) {
    auto input_shape = input.get_legacy_shape();
    auto input_shape_wo_padding = input_shape.without_padding();
    const auto input_rank = input_shape.rank();

    const auto& output_shape = output.get_legacy_shape();
    const auto& output_shape_wo_padding = output_shape.without_padding();
    const auto output_rank = output_shape.rank();

    const bool is_tile_dim = (dim == input_rank - 1 || dim == input_rank - 2);

    log_debug(LogOp, "{}:{} keepdim {} dim {}", __func__, __LINE__, keepdim, dim);
    log_debug(LogOp, "{}:{} input_shape {} wo_padding {}", __func__, __LINE__, input_shape, input_shape_wo_padding);
    log_debug(LogOp, "{}:{} output_shape {} wo_paddoutg {}", __func__, __LINE__, output_shape, output_shape_wo_padding);

    if (keepdim) {
        bool ranks_are_equal = (input_rank == output_rank);
        input_shape[dim] = (is_tile_dim) ? (tt::constants::TILE_HEIGHT) : (1);
        input_shape_wo_padding[dim] = 1;

        if (!ranks_are_equal) {
            log_warning(
                LogOp,
                "{}:{} input_rank {} and output_rank {} are not the same in keepdim mode",
                __func__,
                __LINE__,
                input_rank,
                output_rank);
        }

        std::vector<uint32_t> input_dim(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
        std::vector<uint32_t> output_dim(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
        std::vector<uint32_t> input_dim_wo_padding(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
        std::vector<uint32_t> output_dim_wo_padding(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
        expand_to_max_dim(input_dim, input_shape);
        expand_to_max_dim(output_dim, output_shape);
        expand_to_max_dim(input_dim_wo_padding, input_shape_wo_padding);
        expand_to_max_dim(output_dim_wo_padding, output_shape_wo_padding);

        for (int i = 0; i < input_rank; ++i) {
            TT_FATAL(input_dim[i] == output_dim[i], "Error");
            TT_FATAL(input_dim_wo_padding[i] == output_dim_wo_padding[i], "Error");
        }
    } else {
        TT_FATAL(!is_tile_dim, "Error");
        std::vector<uint32_t> expected_output_shape;
        std::vector<uint32_t> expected_output_shape_wo_padding;
        for (int i = 0; i < output_shape.rank(); ++i) {
            if (i == dim && !is_tile_dim) {
                expected_output_shape.push_back(1);
                expected_output_shape_wo_padding.push_back(1);
            }
            expected_output_shape.push_back(output_shape[i]);
            expected_output_shape_wo_padding.push_back(output_shape_wo_padding[i]);
        }

        log_debug(LogOp, "{}:{} expected_output_shape {}", __func__, __LINE__, expected_output_shape);
        log_debug(
            LogOp, "{}:{} expected_output_shape_wo_padding {}", __func__, __LINE__, expected_output_shape_wo_padding);
        for (int i = 0; i < input_rank; ++i) {
            if (i == dim)
                continue;
            TT_FATAL(input_shape[i] == expected_output_shape[i], "Error");
            TT_FATAL(input_shape_wo_padding[i] == expected_output_shape_wo_padding[i], "Error");
        }
    }
}

}  // namespace

std::tuple<uint32_t, float, bool> get_floored_p_and_decimal_and_p_is_negative(float p) {
    auto floored_p = std::floor(p);
    auto decimal = p - floored_p;
    const bool p_is_negative = floored_p < 0.0f;
    if (p_is_negative) {
        floored_p = -floored_p;
    }
    return std::make_tuple(static_cast<uint32_t>(floored_p), decimal, p_is_negative);
}

void MorehNorm::validate_with_output_tensors(
    const std::vector<Tensor> &input_tensors,
    const std::vector<std::optional<Tensor>> &output_tensors) const {
    const auto &input = input_tensors.at(0);
    const auto &output = output_tensors.at(0);

    check_tensor(input, "moreh_norm", "input");
    check_tensor(output, "moreh_norm", "output");

    validate_input_tensor_with_dim(input, this->dim);

    if (output.has_value()) {
        validate_output_tensor_with_keepdim(input, output.value(), this->dim, this->keepdim);
    }
}

std::vector<tt::tt_metal::LegacyShape> MorehNorm::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    using namespace tt::constants;
    const auto& input = input_tensors.at(0);
    const auto& input_shape = input.get_legacy_shape();
    const auto input_rank = input_shape.rank();
    const bool is_tile_dim = (this->dim == input_rank - 1 || this->dim == input_rank - 2);
    log_debug(LogOp, "{}:{} dim {}, keepdim {}", __func__, __LINE__, this->dim, this->keepdim);

    tt::tt_metal::LegacyShape output_shape = input_shape;
    if (this->keepdim) {
        auto shape = input_shape;
        auto padding = shape.padding();

        if (is_tile_dim) {
            // e.g. (2, 64, 64) with dim 1 to be (2, 1[32], 64)
            shape[this->dim] = TILE_HEIGHT;
            padding[this->dim] = Padding::PadDimension{0, 31};
        } else {
            // e.g. (2, 64, 64) with dim 0 to be (1, 64, 64)
            shape[this->dim] = 1;
        }

        output_shape = tt::tt_metal::LegacyShape(shape, padding);
    } else {
        std::vector<uint32_t> shape;
        std::vector<Padding::PadDimension> pad_dimensions;
        const std::size_t output_rank = (is_tile_dim) ? (input_rank) : (input_rank - 1);
        auto input_padding = input_shape.padding();

        // e.g. (2, 64, 64) with dim 1 to be (2, 1[32], 64)
        // e.g. (2, 64, 64) with dim 0 to be (64, 64)
        for (int i = 0; i < input_rank; ++i) {
            bool is_reduced_dim = (i == this->dim);
            if (is_reduced_dim && !is_tile_dim)
                continue;

            shape.push_back((is_reduced_dim && is_tile_dim) ? (TILE_HEIGHT) : (input_shape[i]));
            pad_dimensions.push_back(
                (is_reduced_dim && is_tile_dim) ? (Padding::PadDimension{0, 31}) : (input_padding[i]));
        }

        auto padding = Padding(pad_dimensions, input_padding.pad_value());
        output_shape = tt::tt_metal::LegacyShape(shape, padding);
    }

    log_debug(LogOp, "{}:{} output_shape {}", __func__, __LINE__, output_shape);
    return {output_shape};
}

std::vector<Tensor> MorehNorm::create_output_tensors(
    const std::vector<Tensor> &input_tensors,
    const std::vector<std::optional<Tensor>>& output_tensors
) const {
    if (output_tensors.at(0).has_value()) {
        return {output_tensors.at(0).value()};
    }

    return operation::generic_create_output_tensors(
        *this, input_tensors, input_tensors.at(0).get_dtype(), Layout::TILE, this->memory_config);
}

Tensor moreh_norm(
    const Tensor &input,
    float p,
    std::optional<std::variant<int64_t, std::vector<int64_t>>> dim,
    const bool keepdim,
    const std::optional<const Tensor> output,
    const std::optional<MemoryConfig> &memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    if (dim == std::nullopt) {
        std::vector<int64_t> dims(input.get_legacy_shape().rank());
        std::iota(dims.begin(), dims.end(), 0);
        dim = std::make_optional(dims);
    }

    if (std::holds_alternative<int64_t>(dim.value())) {
        const auto d = std::get<int64_t>(dim.value());
        return moreh_norm_impl(input, p, d, keepdim, output, memory_config, compute_kernel_config);
    }

    auto dims = std::get<std::vector<int64_t>>(dim.value());

    if (dims.empty()) {
        std::vector<int64_t> all_dims(input.get_legacy_shape().rank());
        std::iota(all_dims.begin(), all_dims.end(), 0);
        dims = all_dims;
    }

    if (dims.size() == 1) {
        const auto d = dims[0];
        return moreh_norm_impl(input, p, d, keepdim, output, memory_config, compute_kernel_config);
    }

    std::sort(dims.begin(), dims.end(), std::greater<int64_t>());
    const auto innermost_dim = dims[0];
    const auto outermost_dim = dims[dims.size() - 1];

    auto tmp_output =
        moreh_norm_impl(input, p, innermost_dim, keepdim, std::nullopt, memory_config, compute_kernel_config);

    using idx_t = decltype(dims.size());
    for (idx_t idx = 1; idx < dims.size() - 1; ++idx) {
        tmp_output =
            moreh_norm_impl(tmp_output, p, dims[idx], keepdim, std::nullopt, memory_config, compute_kernel_config);
    }

    return moreh_norm_impl(
        tmp_output, p, outermost_dim, keepdim, output, memory_config, compute_kernel_config);
}

Tensor moreh_norm_impl(const Tensor &input, float p, int64_t dim,
    const bool keepdim,
    const std::optional<const Tensor> output,
    const std::optional<MemoryConfig> &memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config
) {
    auto device = input.device();

    auto kernel_config_val =
        init_device_compute_kernel_config(device->arch(), compute_kernel_config, MathFidelity::HiFi4);

    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input}))};
    operation::launch_op(
        [p, dim, keepdim, memory_config, kernel_config_val](
            const std::vector<Tensor> &input_tensors,
            const std::vector<std::optional<const Tensor>> &optional_input_tensors,
            const std::vector<std::optional<Tensor>> &optional_output_tensors) mutable -> std::vector<Tensor> {
            return operation::run(
                MorehNorm{
                    .p = p,
                    .dim = dim,
                    .keepdim = keepdim,
                    .memory_config = memory_config.value_or(input_tensors.at(0).memory_config()),
                    .compute_kernel_config = kernel_config_val},
                    input_tensors,
                    optional_input_tensors,
                    optional_output_tensors);
        },
        {input},
        output_tensors,
        {},
        {output});

    return output_tensors.at(0);
}

operation::ProgramWithCallbacks MorehNorm::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto &input = input_tensors.at(0);
    const auto &output = output_tensors.at(0);

    const auto dim = this->dim;
    const auto input_rank = static_cast<decltype(dim)>(input.get_legacy_shape().rank());

    if (dim == input_rank - 1) {
        return moreh_norm_w_impl(input, this->p, output, this->compute_kernel_config);
    } else if (dim == input_rank - 2) {
        return moreh_norm_h_impl(input, this->p, output, this->compute_kernel_config);
    } else {
        return moreh_norm_other_impl(input, this->p, dim, output, this->compute_kernel_config);
    }
}

}  // namespace primary

}  // namespace operations

}  // namespace tt
