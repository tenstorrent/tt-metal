// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "normalization_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"

#include "softmax/softmax_pybind.hpp"
#include "layernorm/layernorm_pybind.hpp"
#include "rmsnorm/rmsnorm_pybind.hpp"
#include "groupnorm/groupnorm_pybind.hpp"
#include "layernorm_distributed/layernorm_distributed_pybind.hpp"
#include "rmsnorm_distributed/rmsnorm_distributed_pybind.hpp"
#include "batch_norm/batch_norm_pybind.hpp"

namespace ttnn::operations::normalization {

void py_module(py::module& module) {
    detail::bind_normalization_softmax(module);
    detail::bind_normalization_layernorm(module);
    detail::bind_normalization_rms_norm(module);
    detail::bind_normalization_group_norm(module);
    detail::bind_normalization_layernorm_distributed(module);
    detail::bind_normalization_rms_norm_distributed(module);
    detail::bind_batch_norm_operation(module);
}

}  // namespace ttnn::operations::normalization
