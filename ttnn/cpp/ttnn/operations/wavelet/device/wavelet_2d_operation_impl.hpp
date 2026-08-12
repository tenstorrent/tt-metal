// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/wavelet/device/wavelet_device_operation_types.hpp"

namespace ttnn::prim::detail {

tt::tt_metal::WorkloadDescriptor create_lwt_2d_workload(
    const Lwt2DParams& operation_attributes,
    const Lwt2DInputs& tensor_args,
    Lwt2DOutputs& tensor_return_value,
    const MeshCoordinateRangeSet& tensor_coords);

void validate_lwt_2d(const Lwt2DParams& operation_attributes, const Lwt2DInputs& tensor_args);
Lwt2DOutputSpecs compute_lwt_2d_output_specs(const Lwt2DParams& operation_attributes, const Lwt2DInputs& tensor_args);
Lwt2DOutputs create_lwt_2d_output_tensors(const Lwt2DParams& operation_attributes, const Lwt2DInputs& tensor_args);

tt::tt_metal::WorkloadDescriptor create_ilwt_2d_workload(
    const Ilwt2DParams& operation_attributes,
    const Ilwt2DInputs& tensor_args,
    Tensor& tensor_return_value,
    const MeshCoordinateRangeSet& tensor_coords);

void validate_ilwt_2d(const Ilwt2DParams& operation_attributes, const Ilwt2DInputs& tensor_args);
tt::tt_metal::TensorSpec compute_ilwt_2d_output_spec(
    const Ilwt2DParams& operation_attributes, const Ilwt2DInputs& tensor_args);
Tensor create_ilwt_2d_output_tensor(const Ilwt2DParams& operation_attributes, const Ilwt2DInputs& tensor_args);

}  // namespace ttnn::prim::detail
