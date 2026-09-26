// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::tensor {

namespace nb = nanobind;

void pytensor_module_types(nb::module_& m_tensor);
void pytensor_module(nb::module_& m_tensor);
void tensor_mem_config_module_types(nb::module_& m_tensor);
void tensor_mem_config_module(nb::module_& m_tensor);

// Registers ttnn.experimental.create_sharded_tensor_view. Takes the experimental module rather
// than the tensor module, so the binding sits with the other tensor bindings while remaining
// reachable only under ttnn.experimental.
void bind_experimental_tensor_view(nb::module_& mod);

}  // namespace ttnn::tensor
