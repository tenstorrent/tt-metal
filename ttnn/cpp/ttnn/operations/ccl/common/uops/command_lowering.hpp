// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/ccl/common/uops/ccl_host_commands.hpp"

#include <vector>

namespace ttnn {
class Tensor;
}  // namespace ttnn

namespace ttnn::ccl {

struct tensor_command_map;
std::vector<ttnn::ccl::cmd::CclHostLowLevelWorkerCommand> tensor_slice_commands_to_noc_commands(
    const std::vector<ttnn::ccl::cmd::CclHostLowLevelWorkerCommand>& command_stream,
    const ttnn::Tensor& tensor,
    size_t packet_size_bytes);
}  // namespace ttnn::ccl
