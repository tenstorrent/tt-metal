// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/wavelet/device/ilwt_1d_program_factory.hpp"

#include "ttnn/operations/wavelet/device/wavelet_1d_operation_impl.hpp"

namespace ttnn::prim {

tt::tt_metal::WorkloadDescriptor Ilwt1DProgramFactory::create_workload_descriptor(
    const Ilwt1DParams& operation_attributes,
    const Ilwt1DInputs& tensor_args,
    Tensor& tensor_return_value,
    const MeshCoordinateRangeSet& tensor_coords) {
    return detail::create_ilwt_1d_workload(operation_attributes, tensor_args, tensor_return_value, tensor_coords);
}

}  // namespace ttnn::prim
