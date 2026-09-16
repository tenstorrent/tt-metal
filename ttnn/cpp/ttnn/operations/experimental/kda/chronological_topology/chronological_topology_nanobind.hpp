// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <nanobind/nanobind.h>
namespace ttnn::operations::experimental::kda::chronological_topology::detail {
void bind_chronological_topology(nanobind::module_&);
}
