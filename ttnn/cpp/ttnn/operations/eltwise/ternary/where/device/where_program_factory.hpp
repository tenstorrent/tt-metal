// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "where_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/cb_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::ternary {
struct WhereProgramFactory {};
}  // namespace ttnn::operations::ternary
