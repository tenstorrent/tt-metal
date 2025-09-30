// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::cluster {
namespace nb = nanobind;

void py_cluster_module_types(nb::module_& mod);
void py_cluster_module(nb::module_& mod);

}  // namespace ttnn::cluster
