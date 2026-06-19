// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "d2d_socket_sync_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct D2DSocketSyncProgramFactory {
    // Per-coord program build. `mesh_dispatch_coordinate` is required: each device's
    // sender service core (data-ready counter, metadata L1, NoC coords) is different,
    // so the per-coord runtime args are looked up by coordinate.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const D2DSocketSyncParams& operation_attributes,
        const D2DSocketSyncInputs& tensor_args,
        Tensor& backing_out,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

}  // namespace ttnn::experimental::prim
