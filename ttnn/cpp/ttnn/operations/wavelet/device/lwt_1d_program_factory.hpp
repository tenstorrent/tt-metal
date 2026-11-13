// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/wavelet/device/wavelet_device_operation_types.hpp"

namespace ttnn::prim {

struct Lwt1DProgramFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const Lwt1DParams& operation_attributes,
        const Lwt1DInputs& tensor_args,
        Lwt1DOutputs& tensor_return_value,
        const MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::prim
