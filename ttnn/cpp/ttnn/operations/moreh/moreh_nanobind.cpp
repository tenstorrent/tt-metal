// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_nanobind.hpp"

#include "ttnn/operations/moreh/moreh_abs_pow/moreh_abs_pow_nanobind.hpp"
#include "ttnn/operations/moreh/moreh_adam/moreh_adam_nanobind.hpp"
#include "ttnn/operations/moreh/moreh_adamw/moreh_adamw_nanobind.hpp"
#include "ttnn/operations/moreh/moreh_arange/moreh_arange_nanobind.hpp"
#include "ttnn/operations/moreh/moreh_bmm/moreh_bmm_nanobind.hpp"
#include "ttnn/operations/moreh/moreh_bmm_backward/moreh_bmm_backward_nanobind.hpp"
#include "ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_nanobind.hpp"
#include "ttnn/operations/moreh/moreh_cumsum/moreh_cumsum_nanobind.hpp"
#include "ttnn/operations/moreh/moreh_dot/moreh_dot_nanobind.hpp"
#include "ttnn/operations/moreh/moreh_dot_backward/moreh_dot_backward_nanobind.hpp"
#include "ttnn/operations/moreh/moreh_fold/fold_nanobind.hpp"
#include "ttnn/operations/moreh/moreh_getitem/moreh_getitem_nanobind.hpp"
#include "ttnn/operations/moreh/moreh_group_norm/moreh_group_norm_nanobind.hpp"
#include "ttnn/operations/moreh/moreh_group_norm_backward/moreh_group_norm_backward_nanobind.hpp"
#include "ttnn/operations/moreh/moreh_layer_norm/moreh_layer_norm_nanobind.hpp"
#include "ttnn/operations/moreh/moreh_layer_norm_backward/moreh_layer_norm_backward_nanobind.hpp"
#include "ttnn/operations/moreh/moreh_linear/moreh_linear_nanobind.hpp"
#include "ttnn/operations/moreh/moreh_linear_backward/moreh_linear_backward_nanobind.hpp"
#include "ttnn/operations/moreh/moreh_matmul/moreh_matmul_nanobind.hpp"
#include "ttnn/operations/moreh/moreh_matmul_backward/moreh_matmul_backward_nanobind.hpp"
#include "ttnn/operations/moreh/moreh_mean/moreh_mean_nanobind.hpp"
#include "ttnn/operations/moreh/moreh_mean_backward/moreh_mean_backward_nanobind.hpp"
#include "ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_nanobind.hpp"
#include "ttnn/operations/moreh/moreh_nll_loss_backward/moreh_nll_loss_backward_nanobind.hpp"
#include "ttnn/operations/moreh/moreh_nll_loss_unreduced_backward/moreh_nll_loss_unreduced_backward_nanobind.hpp"
#include "ttnn/operations/moreh/moreh_norm/moreh_norm_nanobind.hpp"
#include "ttnn/operations/moreh/moreh_norm_backward/moreh_norm_backward_nanobind.hpp"
#include "ttnn/operations/moreh/moreh_sgd/moreh_sgd_nanobind.hpp"
#include "ttnn/operations/moreh/moreh_softmax/moreh_softmax_nanobind.hpp"
#include "ttnn/operations/moreh/moreh_softmax_backward/moreh_softmax_backward_nanobind.hpp"
#include "ttnn/operations/moreh/moreh_sum/moreh_sum_nanobind.hpp"
#include "ttnn/operations/moreh/moreh_sum_backward/moreh_sum_backward_nanobind.hpp"

namespace nb = nanobind;

namespace ttnn::operations::moreh {
void bind_moreh_operations(nb::module_& mod) {
    moreh_abs_pow::bind_moreh_abs_pow_operation(mod);
    moreh_adam::bind_moreh_adam_operation(mod);
    moreh_adamw::bind_moreh_adamw_operation(mod);
    moreh_arange::bind_moreh_arange_operation(mod);
    moreh_bmm_backward::bind_moreh_bmm_backward_operation(mod);
    moreh_bmm::bind_moreh_bmm_operation(mod);
    moreh_cumsum::bind_moreh_cumsum_backward_operation(mod);
    moreh_cumsum::bind_moreh_cumsum_operation(mod);
    moreh_dot_backward::bind_moreh_dot_backward_operation(mod);
    moreh_dot::bind_moreh_dot_operation(mod);
    moreh_fold::bind_moreh_fold_operation(mod);
    moreh_getitem::bind_moreh_getitem_operation(mod);
    moreh_group_norm_backward::bind_moreh_group_norm_backward_operation(mod);
    moreh_group_norm::bind_moreh_group_norm_operation(mod);
    moreh_layer_norm_backward::bind_moreh_layer_norm_backward_operation(mod);
    moreh_layer_norm::bind_moreh_layer_norm_operation(mod);
    moreh_linear_backward::bind_moreh_linear_backward_operation(mod);
    moreh_linear::bind_moreh_linear_operation(mod);
    moreh_matmul_backward::bind_moreh_matmul_backward_operation(mod);
    moreh_matmul::bind_moreh_matmul_operation(mod);
    moreh_mean_backward::bind_moreh_mean_backward_operation(mod);
    moreh_mean::bind_moreh_mean_operation(mod);
    moreh_nll_loss_backward::bind_moreh_nll_loss_backward_operation(mod);
    moreh_nll_loss_unreduced_backward::bind_moreh_nll_loss_unreduced_backward_operation(mod);
    moreh_nll_loss::bind_moreh_nll_loss_operation(mod);
    moreh_norm_backward::bind_moreh_norm_backward_operation(mod);
    moreh_norm::bind_moreh_norm_operation(mod);
    moreh_sgd::bind_moreh_sgd_operation(mod);
    moreh_softmax_backward::bind_moreh_softmax_backward_operation(mod);
    moreh_softmax::bind_moreh_softmax_operation(mod);
    moreh_sum_backward::bind_moreh_sum_backward_operation(mod);
    moreh_sum::bind_moreh_sum_operation(mod);
    moreh_clip_grad_norm::bind_moreh_clip_grad_norm_operation(mod);
}
}  // namespace ttnn::operations::moreh
