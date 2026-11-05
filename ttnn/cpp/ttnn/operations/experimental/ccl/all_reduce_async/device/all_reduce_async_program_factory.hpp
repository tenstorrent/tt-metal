// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "all_reduce_async_device_operation_types.hpp"

#include <tt-metalium/program_descriptors.hpp>

#include <optional>
#include <tuple>
#include <vector>

namespace ttnn::experimental::prim {

struct AllReduceAsyncMeshWorkloadFactory {
    // Program differs per mesh coordinate; this factory does not allocate GlobalSemaphores or Synchronize.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const AllReduceAsyncParams& operation_attributes,
        const AllReduceAsyncInputs& tensor_args,
        Tensor& output_tensor,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);

    // Supersedes automatic buffer-binding patching, so this also refreshes buffer addresses.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const AllReduceAsyncParams& operation_attributes,
        const AllReduceAsyncInputs& tensor_args,
        Tensor& output_tensor,
        const std::optional<ttnn::MeshCoordinate>& coord = std::nullopt);
};

}  // namespace ttnn::experimental::prim

namespace ttnn {

std::tuple<CoreRangeSet, std::vector<CoreCoord>> ar_choose_worker_cores(
    size_t num_links, size_t num_workers_per_link, const CoreRangeSet& available_cores);

}  // namespace ttnn
