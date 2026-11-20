// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "outbound_socket_service_sync_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::experimental::prim {

struct OutboundSocketServiceSyncProgramFactory {
    // One program per chip: each chip's sender service core (data-ready counter, metadata L1, NoC coords) is
    // different. The programs differ only in those runtime args, so a range runs them as one program with
    // per-device runtime args.
    static constexpr bool merge_per_device_programs = true;
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const OutboundSocketServiceSyncParams& args,
        const OutboundSocketServiceSyncInputs& tensor_args,
        Tensor& backing_out,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::experimental::prim
