// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/distributed/types.hpp"
#include "sparse_kv_push_device_operation_types.hpp"
#include "sparse_push_program.hpp"

namespace ttnn::operations::experimental::sparse_kv_push {

sparse_push::PushConfig push_config(const SparseKvPushParams& params, const SparseKvPushInputs& tensor_args);

// One program per chip: routes and the chip's rank depend on its mesh coordinate.
struct SparseKvPushProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const SparseKvPushParams& operation_attributes,
        const SparseKvPushInputs& tensor_args,
        Tensor& output_tensor,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const SparseKvPushParams& operation_attributes,
        const SparseKvPushInputs& tensor_args,
        Tensor& output_tensor,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

}  // namespace ttnn::operations::experimental::sparse_kv_push
