// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/program_descriptors.hpp>

#include "generic_op_device_operation_types.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::generic::program {

struct GenericMeshDescriptorFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);

    // Cache hit: re-apply the coordinate's descriptor (runtime args, named args, tensor-backed CB addresses) to the
    // cached program straight from operation_attributes, instead of the adapter's default of copying the descriptor
    // out of create_descriptor() first. Same writes, no per-call deep copy.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

}  // namespace ttnn::operations::generic::program
