// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"

#include "softmax/softmax_pybind.hpp"
#include "layernorm/layernorm_pybind.hpp"
#include "rmsnorm/rmsnorm_pybind.hpp"
#include "groupnorm/groupnorm_pybind.hpp"

namespace ttnn::operations::normalization {

void py_module(py::module& module) {

    detail::bind_normalization_softmax_program_config_operation(module);
    detail::bind_normalization_softmax_operation(module);
    detail::bind_normalization_scale_mask_softmax_operation(module);
    detail::bind_normalization_softmax_in_place_operation(module);
    detail::bind_normalization_scale_mask_softmax_in_place_operation(module);
    detail::bind_normalization_scale_causal_mask_hw_dims_softmax_in_place_operation(module);
    detail::bind_normalization_layer_norm_operation(module);
    detail::bind_normalization_rms_norm_operation(module);
    detail::bind_normalization_group_norm_operation(module);
}

}  // namespace ttnn::operations::normalization
