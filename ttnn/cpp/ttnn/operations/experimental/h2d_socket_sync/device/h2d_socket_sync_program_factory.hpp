// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>

#include "h2d_socket_sync_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::experimental::prim {

// Per-mesh-coordinate descriptor factory. The 4-arg create_descriptor signature
// (with `mesh_dispatch_coordinate`) makes the device-operation mesh adapter build
// a distinct program per device — needed because each device's service core /
// consumed-counter address differs. See
// ttnn/api/ttnn/mesh_device_operation_adapter.hpp and the dropout mesh factory.
struct H2DSocketSyncProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const H2DSocketSyncParams& args,
        const H2DSocketSyncInputs& tensor_args,
        std::vector<Tensor>& outputs,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

}  // namespace ttnn::experimental::prim
