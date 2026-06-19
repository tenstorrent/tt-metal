// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <tt-metalium/workload_descriptor.hpp>
#include "ttnn/operations/conv/conv2d/device/conv2d_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim::qsr {

struct Conv2dWidthShardedProgramFactory {
    // Builds the workload in one call (cache miss).  The intermediate
    // conv_reader_indices tensor — which must outlive the cached program — is
    // allocated here and parked on the WorkloadDescriptor's `buffers` vector
    // (wrapped in `shared_ptr<Tensor>` so `~Tensor` cannot force-deallocate the
    // device memory while the cached program is still alive).
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const Conv2dParams& operation_attributes,
        const Conv2dInputs& tensor_args,
        Tensor& output_tensor,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::prim::qsr
