// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>

#include "inbound_socket_service_sync_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::experimental::prim {

struct InboundSocketServiceSyncProgramFactory {
    // One program per covered chip: each chip's receiver service core differs. A D2D receiver over a row block of
    // the mesh covers only those rows, so the coordinates it does not cover get no program.
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const InboundSocketServiceSyncParams& args,
        const InboundSocketServiceSyncInputs& tensor_args,
        std::vector<Tensor>& outputs,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::experimental::prim
