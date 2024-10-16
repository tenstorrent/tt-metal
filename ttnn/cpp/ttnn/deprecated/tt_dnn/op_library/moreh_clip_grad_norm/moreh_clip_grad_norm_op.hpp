// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"

namespace tt {

namespace operations {

namespace primary {

using namespace tt_metal;

std::tuple<uint32_t, float, bool> get_p_decimal_p_is_negative(float ord);

struct MorehClipGradNormStep1 {
    float norm_type;
    uint32_t tile_offset_of_tmp_pow_sum;

    void validate(const std::vector<Tensor> &input_tensors,
                  const std::vector<std::optional<const Tensor>> &optional_input_tensors) const;
    std::vector<ttnn::SimpleShape> compute_output_shapes(const std::vector<Tensor> &) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        std::vector<Tensor> &) const;
};

operation::ProgramWithCallbacks moreh_clip_grad_norm_step1_impl(const std::vector<Tensor> &inputs,
                                                                float norm_type,
                                                                uint32_t tile_offset_of_tmp_pow_sum,
                                                                const Tensor &tmp_pow_sum);

void moreh_clip_grad_norm_step1(const std::vector<Tensor> &inputs, float norm_type, const Tensor &tmp_pow_sum);

struct MorehClipGradNormStep2 {
    float norm_type;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<ttnn::SimpleShape> compute_output_shapes(const std::vector<Tensor> &) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor> &input_tensors,
                                                   std::vector<Tensor> &) const;
};

operation::ProgramWithCallbacks moreh_clip_grad_norm_step2_impl(const Tensor &tmp_pow_sum,
                                                                float norm_type,
                                                                const Tensor &total_norm);

void moreh_clip_grad_norm_step2(const Tensor &tmp_pow_sum, float norm_type, const Tensor &total_norm);

struct MorehClipGradNormStep3 {
    void validate(const std::vector<Tensor> &input_tensors,
                  const std::vector<std::optional<const Tensor>> &optional_input_tensors) const;
    std::vector<ttnn::SimpleShape> compute_output_shapes(const std::vector<Tensor> &) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        std::vector<Tensor> &) const;
};

operation::ProgramWithCallbacks moreh_clip_grad_norm_step3_impl(const std::vector<Tensor> &inputs,
                                                                const Tensor &clip_coef_clamped);

void moreh_clip_grad_norm_step3(const std::vector<Tensor> &inputs, const Tensor &clip_coef_clamped);

Tensor moreh_clip_grad_norm_impl(const std::vector<Tensor> &inputs,
                                 float max_norm,
                                 float norm_type,
                                 bool error_if_nonfinite,
                                 const Tensor &tmp_pow_sum,
                                 const Tensor &total_norm);

[[maybe_unused]]
Tensor moreh_clip_grad_norm(const std::vector<Tensor> &inputs,
                            float max_norm,
                            float norm_type,
                            bool error_if_nonfinite,
                            const std::optional<std::reference_wrapper<const Tensor>> total_norm,
                            const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace primary

}  // namespace operations

}  // namespace tt
