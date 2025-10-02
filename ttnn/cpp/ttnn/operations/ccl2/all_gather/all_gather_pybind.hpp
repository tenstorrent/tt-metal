// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file all_gather_pybind.hpp
 * @brief This file contains the Python entry-points for all_gather operation.
 */

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl2/ccl2_common.hpp"
#include "ttnn/operations/ccl2/all_gather/all_gather.hpp"

namespace ttnn::operations::ccl2 {

void py_bind_all_gather(pybind11::module& module);

}  // namespace ttnn::operations::ccl2
