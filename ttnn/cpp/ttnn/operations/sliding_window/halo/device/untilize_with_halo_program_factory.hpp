// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/sliding_window/halo/device/halo_device_operation_types.hpp"

namespace ttnn::prim {

struct UntilizeWithHaloProgramFactory {
    // create_workload_descriptor() allocates the four sliding-window halo
    // config tensors (pad_config0/1, gather_config0/1) on device and parks
    // them on the returned WorkloadDescriptor so their backing buffers
    // outlive the cached programs.  The buffers are bound to the
    // CBDescriptor entries that the reader kernels stream from.
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const HaloParams& operation_attributes,
        const Tensor& tensor_args,
        Tensor& output_tensor,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::prim
