// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/transformer/sdpa/device/sdpa_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"

namespace ttnn::prim {

struct SDPASharedVariables {
    tt::tt_metal::KernelHandle reader_kernels_id{};
    tt::tt_metal::KernelHandle writer_kernels_id{};
    tt::tt_metal::KernelHandle compute_kernels_id{};
    tt::tt_metal::CoreCoord grid_size;
    uint32_t num_cores = 0;
    bool is_chunked = false;
    uint32_t q_chunk_size = 0;
    bool use_mla = false;
};

struct SDPAProgramFactory {
    using shared_variables_t = SDPASharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const SDPAParams& operation_attributes, const SDPAInputs& tensor_args, Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const SDPAParams& operation_attributes,
        const SDPAInputs& tensor_args,
        Tensor& tensor_return_value);
};

// Selected when SDPAParams::mesh_coords is set. Builds a mesh workload that
// only runs the SDPA program on the listed mesh coordinates; other coords are
// left without a program and stay idle for this op.
struct SDPAMeshWorkloadFactory {
    using shared_variables_t = SDPASharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const SDPAParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const SDPAInputs& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const SDPAParams& operation_attributes,
        const SDPAInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
