// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/uops/ccl_command.hpp"
#include "ttnn/operations/ccl/common/uops/ccl_host_commands.hpp"

#include <cstdint>
#include <vector>

namespace ttnn::ccl::worker_detail {

void emit_ccl_send_slice_sequence_commands(std::vector<v1::TensorSlice> const& slices, std::vector<uint32_t>& args_out);

void generate_multi_input_command_stream_kernel_rt_args(
    tt::tt_metal::Program& program,
    tt::tt_metal::KernelHandle kernel_id,
    const Tensor& tensor,
    size_t page_size,
    uint32_t num_pages_per_edm_buffer,  // TODO: get from fabric
    CoreRangeSet const& worker_core_range,
    std::vector<ttnn::ccl::cmd::CclHostLowLevelWorkerCommand> const& ccl_command_stream);
tt::tt_metal::KernelHandle generate_multi_command_stream_kernel_ct_args(
    tt::tt_metal::Program& program,
    const Tensor& tensor,
    const CoreRangeSet& worker_core_range,
    tt::tt_metal::DataMovementConfig datamovement_kernel_config);

}  // namespace ttnn::ccl::worker_detail
