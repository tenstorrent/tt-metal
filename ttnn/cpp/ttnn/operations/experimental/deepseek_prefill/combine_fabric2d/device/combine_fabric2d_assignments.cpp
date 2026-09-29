// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "combine_fabric2d_assignments.hpp"

#include <set>

#include <tt_stl/assert.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

namespace cmbf2d_ns = cmbf2d;

#include "combine_fabric2d_assignments_body.hpp"

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d
