// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>
#include <cstdint>

#include <ttnn/tensor/tensor.hpp>
#include <ttnn/core.hpp>
#include <ttnn/types.hpp>
#include <ttnn/device_operation.hpp>
#include <ttnn/distributed/types.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::debug {

struct ApplyDeviceDelayDeviceOperation {
    struct operation_attributes_t {
        const std::vector<std::vector<uint32_t>> delays;
        const CoreRangeSet worker_core_range_set;
        ttnn::MeshDevice* mesh_device;
    };

    struct tensor_args_t {};

    // Return a minimal dummy tensor since the infrastructure doesn't support void
    using tensor_return_value_t = std::vector<ttnn::Tensor>;
    using spec_return_value_t = std::vector<ttnn::TensorSpec>;

    // Mandatory methods
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    // Per-coord program build.  `mesh_dispatch_coordinate` is required: the delay
    // value baked into the kernel's compile-time args is `delays[row][col]`, so
    // each device in the mesh gets a different program.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

}  // namespace ttnn::operations::debug

namespace ttnn::prim {

ttnn::operations::debug::ApplyDeviceDelayDeviceOperation::tensor_return_value_t apply_device_delay(
    ttnn::MeshDevice& mesh_device,
    const std::vector<std::vector<uint32_t>>& delays,
    const CoreRangeSet& subdevice_core_range_set);

}  // namespace ttnn::prim
