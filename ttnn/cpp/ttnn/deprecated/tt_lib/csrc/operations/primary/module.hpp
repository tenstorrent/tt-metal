// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/deprecated/tt_dnn/op_library/moreh_clip_grad_norm/moreh_clip_grad_norm_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_layernorm/moreh_layernorm_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_layernorm_backward/moreh_layernorm_backward_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_matmul/moreh_matmul_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_matmul_backward/moreh_matmul_backward_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_sum/moreh_sum_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_sum_backward/moreh_sum_backward_op.hpp"

namespace py = pybind11;

namespace tt {
namespace operations {
namespace primary {


void py_module(py::module& m_primary) {
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
}

}  // namespace
   // primary
}  // namespace
   // operations
}  // namespace
   // tt
