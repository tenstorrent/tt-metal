// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "pad_device_operation_types.hpp"

namespace ttnn::prim {

struct PadRmReaderWriterProgramFactory {
    // Workload-scoped pad-value const tensor is allocated once on cache miss inside
    // create_workload_descriptor() and parked on the returned WorkloadDescriptor::buffers
    // so it outlives the cached workload via the program cache.  Holding the SOURCE
    // Tensor (not just shared_ptr<MeshBuffer>) is required because ~Tensor force-deallocates
    // the device memory through DeviceStorage::deallocate regardless of external
    // shared_ptr<MeshBuffer> owners (see #44565).  emplace_runtime_args() with Buffer*
    // lets the framework patch the const tensor's address on cache hits without rerunning
    // this factory.
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const PadParams& operation_attributes,
        const PadInputs& tensor_args,
        Tensor& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};
}  // namespace ttnn::prim
