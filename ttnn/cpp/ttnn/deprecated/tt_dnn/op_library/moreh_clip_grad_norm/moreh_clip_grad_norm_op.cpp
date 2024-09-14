// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <optional>
#include <utility>
#include <vector>

#include "ttnn/deprecated/tt_dnn/op_library/moreh_clip_grad_norm/moreh_clip_grad_norm_op.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"

namespace tt {

namespace operations {

namespace primary {

namespace {

inline uint32_t get_num_device_cores(Device *device) {
    const auto num_cores_x = static_cast<uint32_t>(DeviceComputeWithStorageGridSize(device).x);
    const auto num_cores_y = static_cast<uint32_t>(DeviceComputeWithStorageGridSize(device).y);
    return num_cores_x * num_cores_y;
}
}  // namespace

std::tuple<uint32_t, float, bool> get_p_decimal_p_is_negative(float ord) {
    auto p = std::floor(ord);
    auto decimal = ord - p;
    const bool p_is_negative = p < 0.0f;
    if (p_is_negative) {
        p = -p;
    }
    return std::make_tuple(static_cast<uint32_t>(p), decimal, p_is_negative);
}

void MorehClipGradNormStep1::validate(
    const std::vector<Tensor> &input_tensors,
    const std::vector<std::optional<const Tensor>> &optional_input_tensors) const {
    for (const auto &input : input_tensors) {
        check_tensor(input, "moreh_clip_grad_norm_step1", "input");
    }

    const auto &tmp_pow_sum = optional_input_tensors.at(0).value();
    check_tensor(tmp_pow_sum, "moreh_clip_grad_norm_step1", "tmp_pow_sum");
};

std::vector<Shape> MorehClipGradNormStep1::compute_output_shapes(const std::vector<Tensor> &) const { return {}; }

std::vector<Tensor> MorehClipGradNormStep1::create_output_tensors(const std::vector<Tensor> &) const { return {}; }

operation::ProgramWithCallbacks MorehClipGradNormStep1::create_program(
    const std::vector<Tensor> &input_tensors,
    const std::vector<std::optional<const Tensor>> &optional_input_tensors,
    std::vector<Tensor> &) const {
    const auto &tmp_pow_sum = optional_input_tensors.at(0).value();
    return moreh_clip_grad_norm_step1_impl(
        input_tensors, this->norm_type, this->tile_offset_of_tmp_pow_sum, tmp_pow_sum);
}

void moreh_clip_grad_norm_step1(const std::vector<Tensor> &inputs, float norm_type, const Tensor &tmp_pow_sum) {
    auto device = inputs.at(0).device();
    const auto max_num_inputs = get_num_device_cores(device);
    const auto total_num_inputs = static_cast<uint32_t>(inputs.size());

    const auto num_iter = (total_num_inputs + max_num_inputs - 1) / max_num_inputs;

    uint32_t tile_offset{0};
    auto num_inputs = total_num_inputs;
    for (uint32_t i = 0; i < num_iter; ++i) {
        const auto num_inputs_at_this_iter = std::min(num_inputs, max_num_inputs);

        std::vector<Tensor> dummy_output_tensors = {Tensor(operation::get_workers_for_op_output({tmp_pow_sum}))};

        operation::launch_op(
            [norm_type, tile_offset](
                const std::vector<Tensor> &input_tensors,
                const std::vector<std::optional<const Tensor>> &optional_input_tensors,
                const std::vector<std::optional<Tensor>> &optional_output_tensors) mutable -> std::vector<Tensor> {
                return operation::run(
                    MorehClipGradNormStep1{.norm_type = norm_type, .tile_offset_of_tmp_pow_sum = tile_offset},
                    input_tensors,
                    optional_input_tensors,
                    optional_output_tensors);
            },
            std::vector<Tensor>(inputs.begin() + tile_offset, inputs.begin() + tile_offset + num_inputs_at_this_iter),
            dummy_output_tensors,
            {tmp_pow_sum});

        if (i < (num_iter - 1)) {
            tile_offset += num_inputs_at_this_iter;
            num_inputs -= num_inputs_at_this_iter;
        }
    }
}

void MorehClipGradNormStep2::validate(const std::vector<Tensor> &input_tensors) const {
    const auto &tmp_pow_sum = input_tensors.at(0);
    check_tensor(tmp_pow_sum, "moreh_clip_grad_norm_step2", "tmp_pow_sum");

    const auto &total_norm = input_tensors.at(1);
    check_tensor(total_norm, "moreh_clip_grad_norm_step2", "total_norm");
}

std::vector<Shape> MorehClipGradNormStep2::compute_output_shapes(const std::vector<Tensor> &) const { return {}; }

std::vector<Tensor> MorehClipGradNormStep2::create_output_tensors(const std::vector<Tensor> &) const { return {}; }

operation::ProgramWithCallbacks MorehClipGradNormStep2::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &) const {
    const auto &tmp_pow_sum = input_tensors.at(0);
    const auto &total_norm = input_tensors.at(1);
    return moreh_clip_grad_norm_step2_impl(tmp_pow_sum, this->norm_type, total_norm);
}

void moreh_clip_grad_norm_step2(const Tensor &tmp_pow_sum, float norm_type, const Tensor &total_norm) {
    std::vector<Tensor> dummy_output_tensors = {
        Tensor(operation::get_workers_for_op_output({tmp_pow_sum, total_norm}))};

    operation::launch_op(
        [norm_type](
            const std::vector<Tensor> &input_tensors,
            const std::vector<std::optional<const Tensor>> &optional_input_tensors,
            const std::vector<std::optional<Tensor>> &optional_output_tensors) mutable -> std::vector<Tensor> {
            return operation::run(
                MorehClipGradNormStep2{.norm_type = norm_type},
                input_tensors,
                optional_input_tensors,
                optional_output_tensors);
        },
        {tmp_pow_sum, total_norm},
        dummy_output_tensors);
}

void MorehClipGradNormStep3::validate(
    const std::vector<Tensor> &input_tensors,
    const std::vector<std::optional<const Tensor>> &optional_input_tensors) const {
    for (const auto &input : input_tensors) {
        check_tensor(input, "moreh_clip_grad_norm_step3", "input");
    }

    const auto &clip_coef_clamped = optional_input_tensors.at(0).value();
    check_tensor(clip_coef_clamped, "moreh_clip_grad_norm_step3", "clip_coef_clamped");
}

std::vector<Shape> MorehClipGradNormStep3::compute_output_shapes(const std::vector<Tensor> &) const { return {}; }

std::vector<Tensor> MorehClipGradNormStep3::create_output_tensors(const std::vector<Tensor> &) const { return {}; }

operation::ProgramWithCallbacks MorehClipGradNormStep3::create_program(
    const std::vector<Tensor> &input_tensors,
    const std::vector<std::optional<const Tensor>> &optional_input_tensors,
    std::vector<Tensor> &) const {
    const auto &clip_coef_clamped = optional_input_tensors.at(0).value();
    return moreh_clip_grad_norm_step3_impl(input_tensors, clip_coef_clamped);
}

void moreh_clip_grad_norm_step3(const std::vector<Tensor> &inputs, const Tensor &clip_coef_clamped) {
    auto device = inputs.at(0).device();
    const auto max_num_inputs = get_num_device_cores(device);
    const auto total_num_inputs = static_cast<uint32_t>(inputs.size());

    const auto num_iter = (total_num_inputs + max_num_inputs - 1) / max_num_inputs;

    uint32_t start_input_idx{0};
    auto num_inputs = total_num_inputs;
    for (uint32_t i = 0; i < num_iter; ++i) {
        const auto num_inputs_at_this_iter = std::min(num_inputs, max_num_inputs);

        auto input_tensors = std::vector<Tensor>(
            inputs.begin() + start_input_idx, inputs.begin() + start_input_idx + num_inputs_at_this_iter);
        std::vector<Tensor> dummy_output_tensors = {Tensor(operation::get_workers_for_op_output(input_tensors))};

        operation::launch_op(
            [](const std::vector<Tensor> &input_tensors,
               const std::vector<std::optional<const Tensor>> &optional_input_tensors,
               const std::vector<std::optional<Tensor>> &optional_output_tensors) mutable -> std::vector<Tensor> {
                return operation::run(
                    MorehClipGradNormStep3{}, input_tensors, optional_input_tensors, optional_output_tensors);
            },
            input_tensors,
            dummy_output_tensors,
            {clip_coef_clamped});

        if (i < (num_iter - 1)) {
            start_input_idx += num_inputs_at_this_iter;
            num_inputs -= num_inputs_at_this_iter;
        }
    }
}

Tensor moreh_clip_grad_norm_impl(
    const std::vector<Tensor> &inputs,
    float max_norm,
    float norm_type,
    bool error_if_nonfinite,
    const Tensor &tmp_pow_sum,
    const Tensor &total_norm) {
    // Sum[|e|^p]
    moreh_clip_grad_norm_step1(inputs, norm_type, tmp_pow_sum);

    // Sum[Sum[|e|^p]]^(1/p)
    moreh_clip_grad_norm_step2(tmp_pow_sum, norm_type, total_norm);

    if (error_if_nonfinite) {
        const auto fp32_total_norm =
            tensor_impl::cast_vec<float>(owned_buffer::get_as<bfloat16>(total_norm.cpu())).at(0);
        TT_ASSERT(
            std::isfinite(fp32_total_norm),
            fmt::format(
                "The total norm of order {} for gradients from `parameters` is non-finite, so it cannot be "
                "clipped. To disable this error and scale the gradients by the non-finite norm anyway, set "
                "`error_if_nonfinite=False`",
                norm_type));
    }

    // max_norm / (total_norm + 1e-6)
    const auto &clip_coef = ttnn::multiply(ttnn::add(total_norm, 1e-6f), (1 / max_norm));
    // min(clip_coef, 1.0f)
    Tensor scalar = ttnn::operations::creation::create_scalar(1.0f,inputs.at(0).get_dtype(),Layout::TILE, inputs.at(0).device());
    const auto &clip_coef_clamped = ttnn::minimum(clip_coef, scalar);
    scalar.deallocate();

    // Inplace update inputs(inputs *= clip_coef_clamped)
    moreh_clip_grad_norm_step3(inputs, clip_coef_clamped);

    return total_norm;
}

[[maybe_unused]] Tensor moreh_clip_grad_norm(
    const std::vector<Tensor> &inputs,
    float max_norm,
    float norm_type,
    bool error_if_nonfinite,
    const std::optional<std::reference_wrapper<const Tensor>> total_norm,
    const MemoryConfig &output_mem_config) {
    using namespace tt::constants;
    // Create tmp_pow_sum[1, 1, TILE_HEIGHT, TILE_WIDTH * total_num_inputs]
    const auto total_num_inputs = static_cast<uint32_t>(inputs.size());
    Shape tmp_pow_sum_shape{1, 1, TILE_HEIGHT, TILE_WIDTH * total_num_inputs};
    const auto &tmp_pow_sum =
        create_device_tensor(tmp_pow_sum_shape, inputs.at(0).get_dtype(), Layout::TILE, inputs.at(0).device());

    if (total_norm.has_value() && (total_norm != std::nullopt)) {
        return moreh_clip_grad_norm_impl(
            inputs, max_norm, norm_type, error_if_nonfinite, tmp_pow_sum, total_norm->get());
    }

    // Create total_norm[1, 1, 1, 1]
    Padding padding{{{0, 0}, {0, 0}, {0, TILE_HEIGHT - 1}, {0, TILE_WIDTH - 1}}, Padding::PadValue::Zero};
    Shape total_norm_shape{{1, 1, TILE_HEIGHT, TILE_WIDTH}, padding};
    const auto &created_total_norm = create_device_tensor(
        total_norm_shape, inputs.at(0).get_dtype(), Layout::TILE, inputs.at(0).device(), output_mem_config);

    return moreh_clip_grad_norm_impl(inputs, max_norm, norm_type, error_if_nonfinite, tmp_pow_sum, created_total_norm);
}

}  // namespace primary

}  // namespace operations

}  // namespace tt
