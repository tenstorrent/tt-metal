// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/ccl2/ccl2_common.hpp"
#include "ttnn/operations/ccl2/all_gather/host/all_gather_program.hpp"

namespace ttnn::operations::ccl2 {

struct ExecuteAllGather {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const int32_t dim,
        const ttnn::ccl2::Topology topology,
        const std::optional<ttnn::MemoryConfig>& output_memory_config,
        const std::optional<tt::tt_metal::SubDeviceId> subdevice_id);
};

}  // namespace ttnn::operations::ccl2

namespace ttnn::ccl2 {

constexpr auto all_gather =
    ttnn::register_operation<"ttnn::ccl2::all_gather", ttnn::operations::ccl2::ExecuteAllGather>();

}  // namespace ttnn::ccl2
