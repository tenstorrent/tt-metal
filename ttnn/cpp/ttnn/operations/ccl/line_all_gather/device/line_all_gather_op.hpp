// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "common/core_coord.h"
#include "impl/buffers/buffer.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

#include "ttnn/run_operation.hpp"

#include <optional>
#include <vector>
#include <algorithm>

namespace ttnn {

namespace all_gather_op {
using ccl::Topology;
}; // namespace all_gather_op

using ccl::EriscDatamoverBuilder;

namespace operations {
namespace ccl {

Tensor line_all_gather(
    const Tensor& input_tensor,
    const uint32_t dim,
    const uint32_t num_links = 1,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<size_t> user_defined_num_workers = std::nullopt,
    const std::optional<size_t> user_defined_num_buffers_per_channel = std::nullopt);

Tensor line_all_gather(
    const Tensor& input_tensor,
    const uint32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const uint32_t num_links = 1,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<size_t> user_defined_num_workers = std::nullopt,
    const std::optional<size_t> user_defined_num_buffers_per_channel = std::nullopt);

} // namespace ccl
} // namespace operations

}  // namespace ttnn
