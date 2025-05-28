// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_pybind.hpp"

#include "ttnn/operations/moreh/moreh_abs_pow/moreh_abs_pow_pybind.hpp"
#include "ttnn/operations/moreh/moreh_adam/moreh_adam_pybind.hpp"
#include "ttnn/operations/moreh/moreh_adamw/moreh_adamw_pybind.hpp"
#include "ttnn/operations/moreh/moreh_arange/moreh_arange_pybind.hpp"
#include "ttnn/operations/moreh/moreh_bmm/moreh_bmm_pybind.hpp"
#include "ttnn/operations/moreh/moreh_bmm_backward/moreh_bmm_backward_pybind.hpp"
#include "ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_pybind.hpp"
#include "ttnn/operations/moreh/moreh_cumsum/moreh_cumsum_pybind.hpp"
#include "ttnn/operations/moreh/moreh_dot/moreh_dot_pybind.hpp"
#include "ttnn/operations/moreh/moreh_dot_backward/moreh_dot_backward_pybind.hpp"
#include "ttnn/operations/moreh/moreh_fold/fold_pybind.hpp"
#include "ttnn/operations/moreh/moreh_getitem/moreh_getitem_pybind.hpp"
#include "ttnn/operations/moreh/moreh_group_norm/moreh_group_norm_pybind.hpp"
#include "ttnn/operations/moreh/moreh_group_norm_backward/moreh_group_norm_backward_pybind.hpp"
#include "ttnn/operations/moreh/moreh_layer_norm/moreh_layer_norm_pybind.hpp"
#include "ttnn/operations/moreh/moreh_layer_norm_backward/moreh_layer_norm_backward_pybind.hpp"
#include "ttnn/operations/moreh/moreh_linear/moreh_linear_pybind.hpp"
#include "ttnn/operations/moreh/moreh_linear_backward/moreh_linear_backward_pybind.hpp"
#include "ttnn/operations/moreh/moreh_matmul/moreh_matmul_pybind.hpp"
#include "ttnn/operations/moreh/moreh_matmul_backward/moreh_matmul_backward_pybind.hpp"
#include "ttnn/operations/moreh/moreh_mean/moreh_mean_pybind.hpp"
#include "ttnn/operations/moreh/moreh_mean_backward/moreh_mean_backward_pybind.hpp"
#include "ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_pybind.hpp"
#include "ttnn/operations/moreh/moreh_nll_loss_backward/moreh_nll_loss_backward_pybind.hpp"
#include "ttnn/operations/moreh/moreh_nll_loss_unreduced_backward/moreh_nll_loss_unreduced_backward_pybind.hpp"
#include "ttnn/operations/moreh/moreh_norm/moreh_norm_pybind.hpp"
#include "ttnn/operations/moreh/moreh_norm_backward/moreh_norm_backward_pybind.hpp"
#include "ttnn/operations/moreh/moreh_sgd/moreh_sgd_pybind.hpp"
#include "ttnn/operations/moreh/moreh_softmax/moreh_softmax_pybind.hpp"
#include "ttnn/operations/moreh/moreh_softmax_backward/moreh_softmax_backward_pybind.hpp"
#include "ttnn/operations/moreh/moreh_sum/moreh_sum_pybind.hpp"
#include "ttnn/operations/moreh/moreh_sum_backward/moreh_sum_backward_pybind.hpp"

namespace py = pybind11;

namespace ttnn::operations::moreh {
void bind_moreh_operations(py::module& module) {
    moreh_abs_pow::bind_moreh_abs_pow_operation(module);
    moreh_adam::bind_moreh_adam_operation(module);
    moreh_adamw::bind_moreh_adamw_operation(module);
    moreh_arange::bind_moreh_arange_operation(module);
    moreh_bmm_backward::bind_moreh_bmm_backward_operation(module);
    moreh_bmm::bind_moreh_bmm_operation(module);
    moreh_cumsum::bind_moreh_cumsum_backward_operation(module);
    moreh_cumsum::bind_moreh_cumsum_operation(module);
    moreh_dot_backward::bind_moreh_dot_backward_operation(module);
    moreh_dot::bind_moreh_dot_operation(module);
    moreh_fold::bind_moreh_fold_operation(module);
    moreh_getitem::bind_moreh_getitem_operation(module);
    moreh_group_norm_backward::bind_moreh_group_norm_backward_operation(module);
    moreh_group_norm::bind_moreh_group_norm_operation(module);
    moreh_layer_norm_backward::bind_moreh_layer_norm_backward_operation(module);
    moreh_layer_norm::bind_moreh_layer_norm_operation(module);
    moreh_linear_backward::bind_moreh_linear_backward_operation(module);
    moreh_linear::bind_moreh_linear_operation(module);
    moreh_matmul_backward::bind_moreh_matmul_backward_operation(module);
    moreh_matmul::bind_moreh_matmul_operation(module);
    moreh_mean_backward::bind_moreh_mean_backward_operation(module);
    moreh_mean::bind_moreh_mean_operation(module);
    moreh_nll_loss_backward::bind_moreh_nll_loss_backward_operation(module);
    moreh_nll_loss_unreduced_backward::bind_moreh_nll_loss_unreduced_backward_operation(module);
    moreh_nll_loss::bind_moreh_nll_loss_operation(module);
    moreh_norm_backward::bind_moreh_norm_backward_operation(module);
    moreh_norm::bind_moreh_norm_operation(module);
    moreh_sgd::bind_moreh_sgd_operation(module);
    moreh_softmax_backward::bind_moreh_softmax_backward_operation(module);
    moreh_softmax::bind_moreh_softmax_operation(module);
    moreh_sum_backward::bind_moreh_sum_backward_operation(module);
    moreh_sum::bind_moreh_sum_operation(module);
    moreh_clip_grad_norm::bind_moreh_clip_grad_norm_operation(module);
}
}  // namespace ttnn::operations::moreh
