// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reduction_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/export_enum.hpp"

#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions_pybind.hpp"
#include "ttnn/operations/reduction/argmax/argmax_pybind.hpp"
#include "ttnn/operations/reduction/moe/moe_pybind.hpp"
#include "ttnn/operations/reduction/prod/prod_pybind.hpp"
#include "ttnn/operations/reduction/sampling/sampling_pybind.hpp"
#include "ttnn/operations/reduction/topk/topk_pybind.hpp"

namespace ttnn::operations::reduction {

void py_module(py::module& module) {
    export_enum<ttnn::operations::reduction::ReduceType>(module, "ReduceType");

    // Generic reductions
    detail::bind_reduction_operation(module, ttnn::sum);
    detail::bind_reduction_operation(module, ttnn::mean);
    detail::bind_reduction_operation(module, ttnn::max);
    detail::bind_reduction_operation(module, ttnn::min);
    detail::bind_reduction_operation(module, ttnn::std);
    detail::bind_reduction_operation(module, ttnn::var);

    // Special reductions
    detail::bind_reduction_argmax_operation(module);
    detail::bind_reduction_moe_operation(module);
    detail::bind_reduction_prod_operation(module, ttnn::prod);
    detail::bind_reduction_sampling_operation(module);
    detail::bind_reduction_topk_operation(module);
}

}  // namespace ttnn::operations::reduction
