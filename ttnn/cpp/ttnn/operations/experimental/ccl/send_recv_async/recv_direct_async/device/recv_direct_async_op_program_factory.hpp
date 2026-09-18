// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "recv_direct_async_op_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct RecvDirectAsyncProgramFactory {
    // Per-coord program build. `mesh_dispatch_coordinate` is required: only the devices holding a
    // receiver core of the socket get a program, and their core sets and fabric routes differ, so
    // the descriptor is derived per coordinate. Coordinates with no receiver core yield an empty
    // descriptor, which the framework skips.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const RecvDirectAsyncParams& operation_attributes,
        const Tensor& tensor_args,
        std::vector<Tensor>& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);

    // Re-applies the per-dispatch state on a program-cache hit. The socket config buffer is not
    // reachable from the op's tensors, so it cannot ride the framework's Buffer* binding fast path;
    // this op therefore owns re-applying both it and the output tensor address.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const RecvDirectAsyncParams& operation_attributes,
        const Tensor& tensor_args,
        std::vector<Tensor>& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::experimental::prim
