// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reduction_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/export_enum.hpp"

#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions_nanobind.hpp"
#include "ttnn/operations/reduction/argmax/argmax_nanobind.hpp"
#include "ttnn/operations/reduction/accumulation/cumprod/cumprod_nanobind.hpp"
#include "ttnn/operations/reduction/accumulation/cumsum/cumsum_nanobind.hpp"
#include "ttnn/operations/reduction/moe/moe_nanobind.hpp"
#include "ttnn/operations/reduction/prod/prod_nanobind.hpp"
#include "ttnn/operations/reduction/sampling/sampling_nanobind.hpp"
#include "ttnn/operations/reduction/topk/topk_nanobind.hpp"

namespace ttnn::operations::reduction {

void py_module(nb::module_& mod) {
    export_enum<ttnn::operations::reduction::ReduceType>(mod, "ReduceType");

    // Generic reductions
    detail::bind_reduction_operation(mod, ttnn::sum);
    detail::bind_reduction_operation(mod, ttnn::mean);
    detail::bind_reduction_operation(mod, ttnn::max);
    detail::bind_reduction_operation(mod, ttnn::min);
    detail::bind_reduction_operation(mod, ttnn::std);
    detail::bind_reduction_operation(mod, ttnn::var);

    // Special reductions
    detail::bind_reduction_argmax_operation(mod);
    accumulation::detail::bind_reduction_cumsum_operation(mod);
    accumulation::detail::bind_reduction_cumprod_operation(mod);
    detail::bind_reduction_moe_operation(mod);
    detail::bind_reduction_prod_operation(mod, ttnn::prod);
    detail::bind_reduction_sampling_operation(mod);
    detail::bind_reduction_topk_operation(mod);
}

}  // namespace ttnn::operations::reduction
