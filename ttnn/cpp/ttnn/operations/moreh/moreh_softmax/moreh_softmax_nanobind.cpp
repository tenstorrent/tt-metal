// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_softmax_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "moreh_softmax.hpp"
#include "ttnn-nanobind/decorators.hpp"
#include "ttnn-nanobind/export_enum.hpp"

namespace ttnn::operations::moreh::moreh_softmax {

void bind_moreh_softmax_operation(nb::module_& mod) {
    export_enum<MorehSoftmaxOp>(mod, "MorehSoftmaxOp");
    export_enum<MorehSoftmaxOpParallelizationStrategy>(mod, "MorehSoftmaxOpParallelizationStrategy");

// NOLINTBEGIN(bugprone-macro-parentheses)
#define BIND_MOREH_SOFT_OP(op_name, op_enum, op_desc)                          \
    bind_registered_operation(                                                 \
        mod,                                                                   \
        ttnn::op_name,                                                         \
        op_desc,                                                               \
        ttnn::nanobind_arguments_t{                                            \
            nb::arg("input_tensor"),                                           \
            nb::arg("dim"),                                                    \
            nb::kw_only(),                                                     \
            nb::arg("output_tensor") = nb::none(),                             \
            nb::arg("op") = op_enum,                                           \
            nb::arg("strategy") = MorehSoftmaxOpParallelizationStrategy::NONE, \
            nb::arg("memory_config") = nb::none(),                             \
            nb::arg("compute_kernel_config").noconvert() = nb::none()});
    // NOLINTEND(bugprone-macro-parentheses)

    BIND_MOREH_SOFT_OP(moreh_softmax, MorehSoftmaxOp::SOFTMAX, "Moreh Softmax Operation")
    BIND_MOREH_SOFT_OP(moreh_softmin, MorehSoftmaxOp::SOFTMIN, "Moreh Softmin Operation")
    BIND_MOREH_SOFT_OP(moreh_logsoftmax, MorehSoftmaxOp::LOGSOFTMAX, "Moreh LogSoftmax Operation")
#undef BIND_MOREH_SOFT_OP
}

}  // namespace ttnn::operations::moreh::moreh_softmax
