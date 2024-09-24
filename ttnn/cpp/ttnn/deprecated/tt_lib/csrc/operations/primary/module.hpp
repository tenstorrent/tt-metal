// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/deprecated/tt_dnn/op_library/moreh_adamw/moreh_adamw_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_bmm/moreh_bmm_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_bmm_backward/moreh_bmm_backward_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_clip_grad_norm/moreh_clip_grad_norm_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_cumsum/moreh_cumsum_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_groupnorm/moreh_groupnorm_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_groupnorm_backward/moreh_groupnorm_backward_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_layernorm/moreh_layernorm_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_layernorm_backward/moreh_layernorm_backward_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_linear/moreh_linear_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_linear_backward/moreh_linear_backward_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_matmul/moreh_matmul_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_matmul_backward/moreh_matmul_backward_op.hpp"

#include "ttnn/deprecated/tt_dnn/op_library/moreh_nll_loss/moreh_nll_loss_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_nll_loss_unreduced/moreh_nll_loss_unreduced_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_sgd/moreh_sgd_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_softmax/moreh_softmax_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_softmax_backward/moreh_softmax_backward_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_sum/moreh_sum_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_sum_backward/moreh_sum_backward_op.hpp"

namespace py = pybind11;

namespace tt {
namespace operations {
namespace primary {

void py_module_types(py::module& m_primary) {
    py::enum_<MorehSoftmaxOpParallelizationStrategy>(m_primary, "MorehSoftmaxOpParallelizationStrategy")
        .value("NONE", MorehSoftmaxOpParallelizationStrategy::NONE)
        .value("SMALL_W", MorehSoftmaxOpParallelizationStrategy::SMALL_W)
        .value("SMALL_H", MorehSoftmaxOpParallelizationStrategy::SMALL_H)
        .value("LARGE_W", MorehSoftmaxOpParallelizationStrategy::LARGE_W)
        .value("LARGE_H", MorehSoftmaxOpParallelizationStrategy::LARGE_H)
        .value("LARGE_C", MorehSoftmaxOpParallelizationStrategy::LARGE_C);

    py::enum_<MorehSoftmaxBackwardOpParallelizationStrategy>(m_primary, "MorehSoftmaxBackwardOpParallelizationStrategy")
        .value("NONE", MorehSoftmaxBackwardOpParallelizationStrategy::NONE)
        .value("SMALL_W", MorehSoftmaxBackwardOpParallelizationStrategy::SMALL_W)
        .value("SMALL_H", MorehSoftmaxBackwardOpParallelizationStrategy::SMALL_H)
        .value("LARGE_W", MorehSoftmaxBackwardOpParallelizationStrategy::LARGE_W)
        .value("LARGE_H", MorehSoftmaxBackwardOpParallelizationStrategy::LARGE_H)
        .value("LARGE_C", MorehSoftmaxBackwardOpParallelizationStrategy::LARGE_C);
}

void py_module(py::module& m_primary) {
    // moreh_adamw
    m_primary.def(
        "moreh_adamw",
        &moreh_adamw,
        py::arg("param_in").noconvert(),
        py::arg("grad").noconvert(),
        py::arg("exp_avg_in").noconvert(),
        py::arg("exp_avg_sq_in").noconvert(),
        py::arg("lr").noconvert() = 0.001f,
        py::arg("beta1").noconvert() = 0.9f,
        py::arg("beta2").noconvert() = 0.999f,
        py::arg("eps").noconvert() = 1e-8f,
        py::arg("weight_decay").noconvert() = 0.01f,
        py::arg("step").noconvert(),
        py::arg("amsgrad").noconvert() = false,
        py::arg("max_exp_avg_sq_in").noconvert() = std::nullopt,
        py::arg("param_out").noconvert() = std::nullopt,
        py::arg("exp_avg_out").noconvert() = std::nullopt,
        py::arg("exp_avg_sq_out").noconvert() = std::nullopt,
        py::arg("max_exp_avg_sq_out").noconvert() = std::nullopt,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        R"doc(
        "Performs a moreh_adamw operation.
        )doc");

    // moreh_clip_grad_norm
    m_primary.def(
        "moreh_clip_grad_norm_",
        &moreh_clip_grad_norm,
        py::arg("inputs").noconvert(),
        py::arg("max_norm").noconvert(),
        py::arg("norm_type").noconvert() = 2.0f,
        py::arg("error_if_nonfinite").noconvert() = false,
        py::kw_only(),
        py::arg("total_norm").noconvert() = std::nullopt,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        R"doc(
        "Performs a moreh_clip_grad_norm operation.
    )doc");

    m_primary.def(
        "moreh_bmm",
        &moreh_bmm,
        py::arg("input").noconvert(),
        py::arg("mat2").noconvert(),
        py::kw_only(),
        py::arg("output").noconvert() = std::nullopt,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        R"doc(
        "Performs a moreh_bmm operation.
    )doc");
    m_primary.def(
        "moreh_bmm_backward",
        &moreh_bmm_backward,
        py::arg("output_grad").noconvert(),
        py::arg("input").noconvert(),
        py::arg("mat2").noconvert(),
        py::kw_only(),
        py::arg("are_required_outputs").noconvert() = std::vector<bool>{true, true},
        py::arg("input_grad").noconvert() = std::nullopt,
        py::arg("mat2_grad").noconvert() = std::nullopt,
        py::arg("input_grad_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("mat2_grad_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        R"doc(
        "Performs a moreh_bmm_backward operation.
    )doc");

    m_primary.def(
        "moreh_linear",
        &moreh_linear,
        py::arg("input").noconvert(),
        py::arg("weight").noconvert(),
        py::kw_only(),
        py::arg("bias").noconvert() = std::nullopt,
        py::arg("output").noconvert() = std::nullopt,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        R"doc(
        "Performs a moreh_linear operation.
    )doc");

    m_primary.def(
        "moreh_linear_backward",
        &moreh_linear_backward,
        py::arg("output_grad").noconvert(),
        py::arg("input").noconvert(),
        py::arg("weight").noconvert(),
        py::kw_only(),
        py::arg("are_required_outputs").noconvert() = std::vector<bool>{true, true, true},
        py::arg("bias").noconvert() = std::nullopt,
        py::arg("input_grad").noconvert() = std::nullopt,
        py::arg("weight_grad").noconvert() = std::nullopt,
        py::arg("bias_grad").noconvert() = std::nullopt,
        py::arg("input_grad_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("weight_grad_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("bias_grad_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        R"doc(
        "Performs a moreh_linear_backward operation.
    )doc");

    // moreh_matmul
    m_primary.def(
        "moreh_matmul",
        &moreh_matmul,
        py::arg("input").noconvert(),
        py::arg("other").noconvert(),
        py::kw_only(),
        py::arg("transpose_input") = false,
        py::arg("transpose_other") = false,
        py::arg("output").noconvert() = std::nullopt,
        py::arg("bias").noconvert() = std::nullopt,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        "Performs a moreh_matmul operation.");

    // moreh_matmul_backward
    m_primary.def(
        "moreh_matmul_backward",
        &moreh_matmul_backward,
        py::arg("output_grad").noconvert(),
        py::arg("input_a").noconvert(),
        py::arg("input_b").noconvert(),
        py::arg("are_required_outputs").noconvert() = std::vector<bool>{true, true},
        py::arg("input_a_grad").noconvert() = std::nullopt,
        py::arg("input_b_grad").noconvert() = std::nullopt,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        R"doc(
        "Performs a moreh_matmul_backward operation.
    )doc");

    // moreh_nll_loss
    m_primary.def(
        "moreh_nll_loss",
        &moreh_nll_loss,
        py::arg("input_tensor").noconvert(),
        py::arg("target_tensor").noconvert(),
        py::arg("weight_tensor").noconvert() = std::nullopt,
        py::arg("divisor_tensor").noconvert() = std::nullopt,
        py::arg("output_tensor").noconvert() = std::nullopt,
        py::arg("ignore_index").noconvert(),
        py::arg("reduction_mean").noconvert(),
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        "Performs a nll_loss operation. Returns an output tensor.");

    // moreh_nll_loss_unreduced
    m_primary.def(
        "moreh_nll_loss_unreduced",
        &moreh_nll_loss_unreduced,
        py::arg("input_tensor").noconvert(),
        py::arg("target_tensor").noconvert(),
        py::arg("weight_tensor").noconvert() = std::nullopt,
        py::arg("output_tensor").noconvert() = std::nullopt,
        py::arg("ignore_index").noconvert(),
        py::arg("memory_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        "Performs a nll_loss_unreduced operation. Returns an output tensor.");

    m_primary.def(
        "moreh_layernorm",
        &moreh_layernorm,
        py::arg("input").noconvert(),
        py::arg("normalized_dims").noconvert(),
        py::arg("eps").noconvert() = 1e-5f,
        py::arg("gamma").noconvert() = std::nullopt,
        py::arg("beta").noconvert() = std::nullopt,
        py::kw_only(),
        py::arg("output").noconvert() = std::nullopt,
        py::arg("mean").noconvert() = std::nullopt,
        py::arg("rstd").noconvert() = std::nullopt,
        py::arg("memory_config").noconvert() = std::nullopt,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        "Performs a moreh_layernorm operation.");
    m_primary.def(
        "moreh_layernorm_backward",
        &moreh_layernorm_backward,
        py::arg("output_grad").noconvert(),
        py::arg("input").noconvert(),
        py::arg("mean").noconvert(),
        py::arg("rstd").noconvert(),
        py::arg("normalized_dims").noconvert(),
        py::kw_only(),
        py::arg("gamma").noconvert() = std::nullopt,
        py::arg("input_grad").noconvert() = std::nullopt,
        py::arg("gamma_grad").noconvert() = std::nullopt,
        py::arg("beta_grad").noconvert() = std::nullopt,
        py::arg("memory_config").noconvert() = std::nullopt,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        "Performs a moreh_layernorm_backward operation.");

    m_primary.def(
        "moreh_softmax",
        &moreh_softmax,
        py::arg("input_tensor").noconvert(),
        py::arg("dim").noconvert(),
        py::arg("output_tensor").noconvert() = std::nullopt,
        py::arg("strategy").noconvert() = MorehSoftmaxOpParallelizationStrategy::NONE,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        "Performs a softmax operation. Returns an output tensor.");
    m_primary.def(
        "moreh_softmax_backward",
        &moreh_softmax_backward,
        py::arg("output_tensor").noconvert(),
        py::arg("output_grad_tensor").noconvert(),
        py::arg("dim").noconvert(),
        py::arg("input_grad_tensor").noconvert() = std::nullopt,
        py::arg("strategy").noconvert() = MorehSoftmaxBackwardOpParallelizationStrategy::NONE,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        "Performs a softmax backward operation. Returns an input grad tensor.");
    m_primary.def(
        "moreh_softmin",
        &moreh_softmin,
        py::arg("input_tensor").noconvert(),
        py::arg("dim").noconvert(),
        py::arg("output_tensor").noconvert() = std::nullopt,
        py::arg("strategy").noconvert() = MorehSoftmaxOpParallelizationStrategy::NONE,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        "Performs a softmin operation. Returns an output tensor.");
    m_primary.def(
        "moreh_softmin_backward",
        &moreh_softmin_backward,
        py::arg("output_tensor").noconvert(),
        py::arg("output_grad_tensor").noconvert(),
        py::arg("dim").noconvert(),
        py::arg("input_grad_tensor").noconvert() = std::nullopt,
        py::arg("strategy").noconvert() = MorehSoftmaxBackwardOpParallelizationStrategy::NONE,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        "Performs a softmin backward operation. Returns an input grad tensor.");

    m_primary.def(
        "moreh_logsoftmax",
        &moreh_logsoftmax,
        py::arg("input_tensor").noconvert(),
        py::arg("dim").noconvert(),
        py::arg("output_tensor").noconvert() = std::nullopt,
        py::arg("strategy").noconvert() = MorehSoftmaxOpParallelizationStrategy::NONE,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        "Performs a logsoftmax operation. Returns an output tensor.");

    m_primary.def(
        "moreh_logsoftmax_backward",
        &moreh_logsoftmax_backward,
        py::arg("output_tensor").noconvert(),
        py::arg("output_grad_tensor").noconvert(),
        py::arg("dim").noconvert(),
        py::arg("input_grad_tensor").noconvert() = std::nullopt,
        py::arg("strategy").noconvert() = MorehSoftmaxBackwardOpParallelizationStrategy::NONE,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        "Performs a logsoftmax backward operation. Returns an input grad tensor.");

    m_primary.def(
        "moreh_sum",
        &moreh_sum,
        py::arg("input").noconvert(),
        py::kw_only(),
        py::arg("dim").noconvert() = std::nullopt,
        py::arg("keep_batch_dim").noconvert() = false,
        py::arg("output").noconvert() = std::nullopt,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        py::arg("queue_id").noconvert() = 0,
        "Performs sum operation. Returns an output tensor.");

    m_primary.def(
        "moreh_sum_backward",
        &moreh_sum_backward,
        py::arg("output_grad").noconvert(),
        py::kw_only(),
        py::arg("input").noconvert() = std::nullopt,
        py::arg("dim").noconvert() = std::nullopt,
        py::arg("keep_batch_dim").noconvert() = false,
        py::arg("input_grad").noconvert() = std::nullopt,
        py::arg("input_grad_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        "Performs sum backward operation. Returns an input_grad tensor.");

    m_primary.def(
        "moreh_cumsum",
        &moreh_cumsum,
        py::arg("input").noconvert(),
        py::arg("output").noconvert(),
        py::kw_only(),
        py::arg("dim").noconvert(),
        "Performs cumsum operation. Returns an output tensor.");
    m_primary.def(
        "moreh_cumsum_backward",
        &moreh_cumsum_backward,
        py::arg("output_grad").noconvert(),
        py::arg("input_grad").noconvert(),
        py::kw_only(),
        py::arg("dim").noconvert(),
        "Performs cumsum backward operation. Returns an input_grad tensor.");

    m_primary.def(
        "moreh_sgd",
        &moreh_sgd,
        py::arg("param_in").noconvert(),
        py::arg("grad").noconvert(),
        py::arg("momentum_buffer_in").noconvert() = std::nullopt,
        py::arg("param_out").noconvert() = std::nullopt,
        py::arg("momentum_buffer_out").noconvert() = std::nullopt,
        py::arg("lr").noconvert(),
        py::arg("momentum").noconvert(),
        py::arg("dampening").noconvert(),
        py::arg("weight_decay").noconvert(),
        py::arg("nesterov").noconvert(),
        py::arg("momentum_initialized").noconvert(),
        py::arg("param_out_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("momentum_buffer_out_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("compute_kernel_config").noconvert() = std::nullopt,
        "Performs a SGD operation.");

    // moreh_groupnorm
    m_primary.def(
        "moreh_groupnorm",
        &moreh_groupnorm,
        py::arg("input").noconvert(),
        py::arg("num_groups").noconvert(),
        py::arg("eps").noconvert() = 1e-5f,
        py::arg("gamma").noconvert() = std::nullopt,
        py::arg("beta").noconvert() = std::nullopt,
        py::kw_only(),
        py::arg("are_required_outputs").noconvert() = std::vector<bool>{true, false, false},
        py::arg("output").noconvert() = std::nullopt,
        py::arg("mean").noconvert() = std::nullopt,
        py::arg("rstd").noconvert() = std::nullopt,
        py::arg("output_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("mean_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("rstd_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        R"doc(
        Performs a moreh_groupnorm operation.
    )doc");

    // moreh_groupnorm_backward
    m_primary.def(
        "moreh_groupnorm_backward",
        &moreh_groupnorm_backward,
        py::arg("output_grad").noconvert(),
        py::arg("input").noconvert(),
        py::arg("mean").noconvert(),
        py::arg("rstd").noconvert(),
        py::arg("num_groups").noconvert(),
        py::kw_only(),
        py::arg("are_required_outputs").noconvert() = std::vector<bool>{true, true, true},
        py::arg("gamma").noconvert() = std::nullopt,
        py::arg("input_grad").noconvert() = std::nullopt,
        py::arg("gamma_grad").noconvert() = std::nullopt,
        py::arg("beta_grad").noconvert() = std::nullopt,
        py::arg("input_grad_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("gamma_grad_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("beta_grad_mem_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        R"doc(
        Performs a moreh_groupnorm_backward operation.
    )doc");


}

}  // namespace
   // primary
}  // namespace
   // operations
}  // namespace
   // tt
