// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/distributed/types.hpp"
#include "sparse_kv_dispatch_device_operation_types.hpp"

namespace ttnn::operations::experimental::sparse_kv_push {

struct SparseKvDispatchProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const SparseKvDispatchParams& params,
        const SparseKvDispatchInputs& tensor_args,
        Tensor& output_tensor,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const SparseKvDispatchParams& params,
        const SparseKvDispatchInputs& tensor_args,
        Tensor& output_tensor,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

}  // namespace ttnn::operations::experimental::sparse_kv_push
