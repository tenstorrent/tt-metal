// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "send_async_op_device_operation_types.hpp"

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>
#include "ttnn/distributed/types.hpp"

namespace ttnn::experimental::prim {

struct SendAsyncMeshWorkloadFactory {
    // Contract (2): declarative WorkloadDescriptor.
    //
    // For coordinates that are not sender devices for the socket, an empty
    // ProgramDescriptor (noop=true) is emitted: no kernels, no CBs.  The
    // legacy create_mesh_workload pre-filtered such coords out via
    // get_workload_coords; in the descriptor pattern the framework iterates
    // every tensor coord, so we emit a no-op program for non-sender coords.
    //
    // The MeshSocket lives on SendAsyncParams (caller allocated). The reader's
    // input_tensor address is patched via BufferBinding on cache hit; the writer's
    // socket-config-buffer address is workload-scoped (stable across dispatches)
    // so it stays as a raw uint32_t — no rebuild path is available in contract-2.
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const SendAsyncParams& operation_attributes,
        const Tensor& tensor_args,
        std::vector<Tensor>& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::experimental::prim
