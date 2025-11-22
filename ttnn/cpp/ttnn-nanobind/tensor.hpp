// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::tensor {

namespace nb = nanobind;

void pytensor_module_types(nb::module_& m_tensor);
void pytensor_module(nb::module_& m_tensor);
void tensor_mem_config_module_types(nb::module_& mod);
void tensor_mem_config_module(nb::module_& mod);

}  // namespace ttnn::tensor
