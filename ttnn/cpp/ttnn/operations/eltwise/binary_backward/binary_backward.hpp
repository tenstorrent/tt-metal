
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/common/constants.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/eltwise/complex_binary/device/complex_binary_op.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"

namespace ttnn {

namespace operations::binary_backward {

struct ExecuteBackwardAtan2 {
    static std::vector<Tensor> invoke(const Tensor &grad_tensor_arg,
                                      const Tensor &input_tensor_a_arg,
                                      const Tensor &input_tensor_b_arg,
                                      const std::optional<MemoryConfig> &memory_config = std::nullopt);
};

struct ExecuteBackwardXlogy {
    static std::vector<Tensor> invoke(const Tensor &grad_tensor_arg,
                                      const Tensor &input_tensor_a_arg,
                                      const Tensor &input_tensor_b_arg,
                                      const std::optional<MemoryConfig> &memory_config = std::nullopt);
};

struct ExecuteBackwardHypot {
    static std::vector<Tensor> invoke(const Tensor &grad_tensor_arg,
                                      const Tensor &input_tensor_a_arg,
                                      const Tensor &input_tensor_b_arg,
                                      const std::optional<MemoryConfig> &memory_config = std::nullopt);
};

struct ExecuteBackwardLdexp {
    static std::vector<Tensor> invoke(const Tensor &grad_tensor_arg,
                                      const Tensor &input_tensor_a_arg,
                                      const Tensor &input_tensor_b_arg,
                                      const std::optional<MemoryConfig> &memory_config = std::nullopt);
};

struct ExecuteBackwardLogaddexp {
    static std::vector<Tensor> invoke(const Tensor &grad_tensor_arg,
                                      const Tensor &input_tensor_a_arg,
                                      const Tensor &input_tensor_b_arg,
                                      const std::optional<MemoryConfig> &memory_config = std::nullopt);
};

struct ExecuteBackwardLogaddexp2 {
    static std::vector<Tensor> invoke(const Tensor &grad_tensor_arg,
                                      const Tensor &input_tensor_a_arg,
                                      const Tensor &input_tensor_b_arg,
                                      const std::optional<MemoryConfig> &memory_config = std::nullopt);
};

struct ExecuteBackwardSquaredDifference {
    static std::vector<Tensor> invoke(const Tensor &grad_tensor_arg,
                                      const Tensor &input_tensor_a_arg,
                                      const Tensor &input_tensor_b_arg,
                                      const std::optional<MemoryConfig> &memory_config = std::nullopt);
};

struct ExecuteBackwardMax {
    static std::vector<Tensor> invoke(const Tensor &grad_tensor_arg,
                                      const Tensor &input_tensor_a_arg,
                                      const Tensor &input_tensor_b_arg,
                                      const std::optional<MemoryConfig> &memory_config = std::nullopt);
};

struct ExecuteBackwardMin {
    static std::vector<Tensor> invoke(const Tensor &grad_tensor_arg,
                                      const Tensor &input_tensor_a_arg,
                                      const Tensor &input_tensor_b_arg,
                                      const std::optional<MemoryConfig> &memory_config = std::nullopt);
};

struct ExecuteBackwardMul {
    static std::vector<std::optional<ttnn::Tensor>> invoke(
        uint8_t queue_id,
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        float scalar,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt);

    static std::vector<std::optional<Tensor>> invoke(
        uint8_t queue_id,
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        const Tensor &other_tensor_arg,
        const std::vector<bool> &are_required_outputs = std::vector<bool>{true, true},
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt,
        std::optional<Tensor> other_grad = std::nullopt);

    static std::vector<std::optional<ttnn::Tensor>> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        float scalar,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt);

    static std::vector<std::optional<Tensor>> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        const Tensor &other_tensor_arg,
        const std::vector<bool> &are_required_outputs = std::vector<bool>{true, true},
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt,
        std::optional<Tensor> other_grad = std::nullopt);

    static std::vector<ComplexTensor> invoke(const ComplexTensor &grad_tensor_arg,
                                             const ComplexTensor &input_tensor_a_arg,
                                             const ComplexTensor &input_tensor_b_arg,
                                             const MemoryConfig &memory_config);
};

struct ExecuteBackwardAssign {
    static std::vector<std::optional<ttnn::Tensor>> invoke(
        uint8_t queue_id,
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> input_a_grad = std::nullopt);

    static std::vector<std::optional<ttnn::Tensor>> invoke(
        uint8_t queue_id,
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        const Tensor &other_tensor_arg,
        const std::vector<bool> &are_required_outputs = std::vector<bool>{true, true},
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> input_a_grad = std::nullopt,
        std::optional<Tensor> input_b_grad = std::nullopt);

    static std::vector<std::optional<ttnn::Tensor>> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> input_a_grad = std::nullopt);

    static std::vector<std::optional<ttnn::Tensor>> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        const Tensor &other_tensor_arg,
        const std::vector<bool> &are_required_outputs = std::vector<bool>{true, true},
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> input_a_grad = std::nullopt,
        std::optional<Tensor> input_b_grad = std::nullopt);
};

struct ExecuteBackwardBiasGelu {
    static std::vector<Tensor> invoke(const Tensor &grad_tensor_arg,
                                      const Tensor &input_tensor_a_arg,
                                      const Tensor &input_tensor_b_arg,
                                      string approximate,
                                      const std::optional<MemoryConfig> &memory_config = std::nullopt);

    static std::vector<Tensor> invoke(const Tensor &grad_tensor_arg,
                                      const Tensor &input_tensor_a_arg,
                                      float scalar,
                                      string approximate,
                                      const std::optional<MemoryConfig> &memory_config = std::nullopt);
};

struct ExecuteBackwardLT {
    static std::vector<std::optional<Tensor>> invoke(
        uint8_t queue_id,
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        const Tensor &other_tensor_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        const std::vector<bool> &are_required_outputs = std::vector<bool>{true, true},
        std::optional<Tensor> input_grad = std::nullopt,
        std::optional<Tensor> other_grad = std::nullopt);

    static std::vector<std::optional<Tensor>> invoke(uint8_t queue_id,
                                                     const Tensor &grad_tensor_arg,
                                                     const Tensor &input_tensor_arg,
                                                     float other,
                                                     const std::optional<MemoryConfig> &memory_config = std::nullopt,
                                                     std::optional<Tensor> input_grad = std::nullopt);

    static std::vector<std::optional<Tensor>> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        const Tensor &other_tensor_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        const std::vector<bool> &are_required_outputs = std::vector<bool>{true, true},
        std::optional<Tensor> input_grad = std::nullopt,
        std::optional<Tensor> other_grad = std::nullopt);

    static std::vector<std::optional<Tensor>> invoke(const Tensor &grad_tensor_arg,
                                                     const Tensor &input_tensor_arg,
                                                     float other,
                                                     const std::optional<MemoryConfig> &memory_config = std::nullopt,
                                                     std::optional<Tensor> input_grad = std::nullopt);
};

struct ExecuteBackwardAdd {
    static std::vector<std::optional<Tensor>> invoke(uint8_t queue_id,
                                                     const Tensor &grad_tensor_arg,
                                                     const Tensor &input_tensor_arg,
                                                     float scalar,
                                                     const std::optional<MemoryConfig> &memory_config = std::nullopt,
                                                     std::optional<Tensor> input_grad = std::nullopt);

    static std::vector<std::optional<Tensor>> invoke(
        uint8_t queue_id,
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const std::vector<bool> &are_required_outputs = std::vector<bool>{true, true},
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt,
        std::optional<Tensor> other_grad = std::nullopt);

    static std::vector<std::optional<Tensor>> invoke(const Tensor &grad_tensor_arg,
                                                     const Tensor &input_tensor_arg,
                                                     float scalar,
                                                     const std::optional<MemoryConfig> &memory_config = std::nullopt,
                                                     std::optional<Tensor> input_grad = std::nullopt);

    static std::vector<std::optional<Tensor>> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const std::vector<bool> &are_required_outputs = std::vector<bool>{true, true},
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt,
        std::optional<Tensor> other_grad = std::nullopt);

    static std::vector<ComplexTensor> invoke(const ComplexTensor &grad_tensor_arg,
                                             const ComplexTensor &input_tensor_a_arg,
                                             const ComplexTensor &input_tensor_b_arg,
                                             float alpha,
                                             const std::optional<MemoryConfig> &memory_config = std::nullopt);
};

struct ExecuteBackwardSub {
    static std::vector<std::optional<Tensor>> invoke(uint8_t queue_id,
                                                     const Tensor &grad_tensor_arg,
                                                     const Tensor &input_tensor_arg,
                                                     float scalar,
                                                     const std::optional<MemoryConfig> &memory_config = std::nullopt,
                                                     std::optional<Tensor> input_grad = std::nullopt);

    static std::vector<std::optional<Tensor>> invoke(
        uint8_t queue_id,
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const std::vector<bool> &are_required_outputs = std::vector<bool>{true, true},
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt,
        std::optional<Tensor> other_grad = std::nullopt);

    static std::vector<std::optional<Tensor>> invoke(const Tensor &grad_tensor_arg,
                                                     const Tensor &input_tensor_arg,
                                                     float scalar,
                                                     const std::optional<MemoryConfig> &memory_config = std::nullopt,
                                                     std::optional<Tensor> input_grad = std::nullopt);

    static std::vector<std::optional<Tensor>> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const std::vector<bool> &are_required_outputs = std::vector<bool>{true, true},
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt,
        std::optional<Tensor> other_grad = std::nullopt);

    static std::vector<ComplexTensor> invoke(const ComplexTensor &grad_tensor_arg,
                                             const ComplexTensor &input_tensor_a_arg,
                                             const ComplexTensor &input_tensor_b_arg,
                                             float alpha,
                                             const std::optional<MemoryConfig> &memory_config = std::nullopt);
};

struct ExecuteBackwardDiv {
    static std::vector<std::optional<Tensor>> invoke(uint8_t queue_id,
                                                     const Tensor &grad_tensor_arg,
                                                     const Tensor &input_tensor_arg,
                                                     float scalar,
                                                     string round_mode = "None",
                                                     const std::optional<MemoryConfig> &memory_config = std::nullopt,
                                                     std::optional<Tensor> input_grad = std::nullopt);

    static std::vector<std::optional<Tensor>> invoke(
        uint8_t queue_id,
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        const Tensor &other_tensor_arg,
        string round_mode = "None",
        const std::vector<bool> &are_required_outputs = std::vector<bool>{true, true},
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt,
        std::optional<Tensor> other_grad = std::nullopt);

    static std::vector<std::optional<Tensor>> invoke(const Tensor &grad_tensor_arg,
                                                     const Tensor &input_tensor_arg,
                                                     float scalar,
                                                     string round_mode = "None",
                                                     const std::optional<MemoryConfig> &memory_config = std::nullopt,
                                                     std::optional<Tensor> input_grad = std::nullopt);

    static std::vector<std::optional<Tensor>> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        const Tensor &other_tensor_arg,
        string round_mode = "None",
        const std::vector<bool> &are_required_outputs = std::vector<bool>{true, true},
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt,
        std::optional<Tensor> other_grad = std::nullopt);

    static std::vector<ComplexTensor> invoke(const ComplexTensor &grad_tensor_arg,
                                             const ComplexTensor &input_tensor_a_arg,
                                             const ComplexTensor &input_tensor_b_arg,
                                             const MemoryConfig &memory_config);
};

struct ExecuteBackwardRemainder {
    static std::vector<Tensor> invoke(const Tensor &grad_tensor_arg,
                                      const Tensor &input_tensor_arg,
                                      float scalar,
                                      const std::optional<MemoryConfig> &memory_config = std::nullopt);

    static std::vector<Tensor> invoke(const Tensor &grad_tensor_arg,
                                      const Tensor &input_tensor_a_arg,
                                      const Tensor &input_tensor_b_arg,
                                      const std::optional<MemoryConfig> &memory_config = std::nullopt);
};

struct ExecuteBackwardFmod {
    static std::vector<Tensor> invoke(const Tensor &grad_tensor_arg,
                                      const Tensor &input_tensor_arg,
                                      float scalar,
                                      const std::optional<MemoryConfig> &memory_config = std::nullopt);

    static std::vector<Tensor> invoke(const Tensor &grad_tensor_arg,
                                      const Tensor &input_tensor_a_arg,
                                      const Tensor &input_tensor_b_arg,
                                      const std::optional<MemoryConfig> &memory_config = std::nullopt);
};

struct ExecuteAddalphaBW {
    static std::vector<std::optional<Tensor>> invoke(
        uint8_t queue_id,
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        float parameter,
        const std::vector<bool> &are_required_outputs = std::vector<bool>{true, true},
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> input_a_grad = std::nullopt,
        std::optional<Tensor> input_b_grad = std::nullopt);

    static std::vector<std::optional<Tensor>> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        float parameter,
        const std::vector<bool> &are_required_outputs = std::vector<bool>{true, true},
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> input_a_grad = std::nullopt,
        std::optional<Tensor> input_b_grad = std::nullopt);
};

struct ExecuteBackwardSubAlpha {
    static std::vector<std::optional<ttnn::Tensor>> invoke(
        uint8_t queue_id,
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        float alpha,
        const std::vector<bool> &are_required_outputs = std::vector<bool>{true, true},
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt,
        std::optional<Tensor> other_grad = std::nullopt);

    static std::vector<std::optional<ttnn::Tensor>> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        float alpha,
        const std::vector<bool> &are_required_outputs = std::vector<bool>{true, true},
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt,
        std::optional<Tensor> other_grad = std::nullopt);
};

struct ExecuteBackwardRsub {
    static std::vector<std::optional<ttnn::Tensor>> invoke(
        uint8_t queue_id,
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const std::vector<bool> &are_required_outputs = std::vector<bool>{true, true},
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt,
        std::optional<Tensor> other_grad = std::nullopt);

    static std::vector<std::optional<ttnn::Tensor>> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const std::vector<bool> &are_required_outputs = std::vector<bool>{true, true},
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt,
        std::optional<Tensor> other_grad = std::nullopt);
};

struct ExecuteBackwardConcat {
    static std::vector<std::optional<Tensor>> invoke(
        uint8_t queue_id,
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        int dim,
        const std::vector<bool> &are_required_outputs = std::vector<bool>{true, true},
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt,
        std::optional<Tensor> other_grad = std::nullopt);

    static std::vector<std::optional<Tensor>> invoke(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        int dim,
        const std::vector<bool> &are_required_outputs = std::vector<bool>{true, true},
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt,
        std::optional<Tensor> other_grad = std::nullopt);
};

}  // namespace operations::binary_backward

constexpr auto atan2_bw =
    ttnn::register_operation<"ttnn::atan2_bw", operations::binary_backward::ExecuteBackwardAtan2>();
constexpr auto xlogy_bw =
    ttnn::register_operation<"ttnn::xlogy_bw", operations::binary_backward::ExecuteBackwardXlogy>();
constexpr auto hypot_bw =
    ttnn::register_operation<"ttnn::hypot_bw", operations::binary_backward::ExecuteBackwardHypot>();
constexpr auto ldexp_bw =
    ttnn::register_operation<"ttnn::ldexp_bw", operations::binary_backward::ExecuteBackwardLdexp>();
constexpr auto logaddexp_bw =
    ttnn::register_operation<"ttnn::logaddexp_bw", operations::binary_backward::ExecuteBackwardLogaddexp>();
constexpr auto logaddexp2_bw =
    ttnn::register_operation<"ttnn::logaddexp2_bw", operations::binary_backward::ExecuteBackwardLogaddexp2>();
constexpr auto squared_difference_bw =
    ttnn::register_operation<"ttnn::squared_difference_bw",
                             operations::binary_backward::ExecuteBackwardSquaredDifference>();
constexpr auto min_bw = ttnn::register_operation<"ttnn::min_bw", operations::binary_backward::ExecuteBackwardMin>();
constexpr auto max_bw = ttnn::register_operation<"ttnn::max_bw", operations::binary_backward::ExecuteBackwardMax>();

constexpr auto subalpha_bw =
    ttnn::register_operation<"ttnn::subalpha_bw", operations::binary_backward::ExecuteBackwardSubAlpha>();

constexpr auto rsub_bw = ttnn::register_operation<"ttnn::rsub_bw", operations::binary_backward::ExecuteBackwardRsub>();

constexpr auto concat_bw =
    ttnn::register_operation<"ttnn::concat_bw", operations::binary_backward::ExecuteBackwardConcat>();

constexpr auto mul_bw = ttnn::register_operation<"ttnn::mul_bw", operations::binary_backward::ExecuteBackwardMul>();

constexpr auto assign_bw =
    ttnn::register_operation<"ttnn::assign_bw", operations::binary_backward::ExecuteBackwardAssign>();

constexpr auto bias_gelu_bw =
    ttnn::register_operation<"ttnn::bias_gelu_bw", operations::binary_backward::ExecuteBackwardBiasGelu>();

constexpr auto addalpha_bw =
    ttnn::register_operation<"ttnn::addalpha_bw", operations::binary_backward::ExecuteAddalphaBW>();

constexpr auto add_bw = ttnn::register_operation<"ttnn::add_bw", operations::binary_backward::ExecuteBackwardAdd>();

constexpr auto sub_bw = ttnn::register_operation<"ttnn::sub_bw", operations::binary_backward::ExecuteBackwardSub>();

constexpr auto div_bw = ttnn::register_operation<"ttnn::div_bw", operations::binary_backward::ExecuteBackwardDiv>();

constexpr auto remainder_bw =
    ttnn::register_operation<"ttnn::remainder_bw", operations::binary_backward::ExecuteBackwardRemainder>();

constexpr auto fmod_bw = ttnn::register_operation<"ttnn::fmod_bw", operations::binary_backward::ExecuteBackwardFmod>();

}  // namespace ttnn
