// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/cpp/ttnn/operations/ccl/common/uops/ccl_command.hpp"

#include <ranges>


namespace ttnn {
namespace ccl {
namespace cmd {


std::vector<uint32_t> add_ccl_command_to_args(CclCommand const& cmd);




} // namespace cmd
} // namespace ccl
} // namespace ttnn
