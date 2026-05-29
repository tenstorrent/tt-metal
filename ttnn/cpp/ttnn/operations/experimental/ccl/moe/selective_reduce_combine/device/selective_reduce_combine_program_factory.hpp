// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "selective_reduce_combine_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

namespace ttnn::experimental::prim {

struct SelectiveReduceCombineProgramArtifacts {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    tt::tt_metal::CBHandle data_cb_handle{};
    std::vector<tt::tt_metal::CoreCoord> cores;
    const GlobalSemaphore init_semaphore;
    const GlobalSemaphore cross_device_semaphore;
};

// Descriptor flavor of SelectiveReduceCombineProgramArtifacts.
// Returned by build_selective_reduce_combine_program_artifacts_descriptor below.
// Holds indices into the caller's ProgramDescriptor::kernels for the reader and
// writer kernels — descriptor-based callers don't need a separate CBHandle
// because the input-tensor binding lives on the CBDescriptor (via its buffer
// pointer) and is patched by the framework's fast cache-hit path.
struct SelectiveReduceCombineProgramArtifactsDescriptor {
    tt::tt_metal::KernelHandle reader_kernel_index{};
    tt::tt_metal::KernelHandle writer_kernel_index{};
    std::vector<tt::tt_metal::CoreCoord> cores;
    const GlobalSemaphore init_semaphore;
    const GlobalSemaphore cross_device_semaphore;
};

struct UnifiedSelectReduce {
    using operation_attributes_t = SelectiveReduceCombineParams;
    using tensor_args_t = SelectiveReduceCombineTensors;
    using tensor_return_value_t = ttnn::Tensor;

    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

// Builder function that creates kernels and returns artifacts
SelectiveReduceCombineProgramArtifacts build_selective_reduce_combine_program_artifacts(
    tt::tt_metal::Program& program,
    const experimental::prim::SelectiveReduceCombineParams& operation_attributes,
    const MeshCoordinate& mesh_coordinate,
    const std::vector<MeshCoordinate>& all_mesh_coordinates,
    const experimental::prim::SelectiveReduceCombineTensors& tensor_args,
    Tensor& tensor_return_value,
    const GlobalSemaphore& init_semaphore,
    const GlobalSemaphore& cross_device_semaphore,
    const uint32_t metadata_sync_semaphore_id,
    const uint32_t compute_sync_semaphore_id,
    const uint32_t compute_cores_per_combine_cores = 0,
    const std::optional<std::vector<CoreCoord>>& compute_cores_by_ring_id = std::nullopt);

// ProgramDescriptor variant of build_selective_reduce_combine_program_artifacts.
// Appends kernels / CBs / semaphores onto the caller's ProgramDescriptor instead
// of issuing imperative CreateKernel / CreateCircularBuffer / CreateSemaphore /
// SetRuntimeArgs calls.  The input tensor binding for the data CB lives on
// CBDescriptor::buffer (instead of being patched via UpdateDynamicCircularBufferAddress)
// and the buffer addresses for the reader and writer runtime args are bound via
// emplace_runtime_args so the framework's fast cache-hit path patches them without
// rebuilding the program.  metadata and compute sync semaphore IDs are still passed
// in by the caller (allocated by pushing SemaphoreDescriptors onto desc.semaphores
// prior to invoking this helper) so the helper does not need to know the caller's
// other semaphore IDs.
SelectiveReduceCombineProgramArtifactsDescriptor build_selective_reduce_combine_program_artifacts_descriptor(
    tt::tt_metal::ProgramDescriptor& desc,
    const experimental::prim::SelectiveReduceCombineParams& operation_attributes,
    const MeshCoordinate& mesh_coordinate,
    const std::vector<MeshCoordinate>& all_mesh_coordinates,
    const experimental::prim::SelectiveReduceCombineTensors& tensor_args,
    Tensor& tensor_return_value,
    const GlobalSemaphore& init_semaphore,
    const GlobalSemaphore& cross_device_semaphore,
    const uint32_t metadata_sync_semaphore_id,
    const uint32_t compute_sync_semaphore_id,
    const uint32_t compute_cores_per_combine_cores = 0,
    const std::optional<std::vector<CoreCoord>>& compute_cores_by_ring_id = std::nullopt);

// Runtime argument override function
void selective_reduce_combine_helper_override_runtime_arguments(
    tt::tt_metal::Program& program,
    tt::tt_metal::KernelHandle reader_kernel_id,
    tt::tt_metal::KernelHandle writer_kernel_id,
    tt::tt_metal::CBHandle data_cb_handle,
    const std::vector<tt::tt_metal::CoreCoord>& cores,
    const experimental::prim::SelectiveReduceCombineTensors& tensor_args,
    Tensor& tensor_return_value,
    const GlobalSemaphore& init_semaphore,
    const GlobalSemaphore& cross_device_semaphore,
    const std::optional<GlobalSemaphore>& optional_cross_device_semaphore);

}  // namespace ttnn::experimental::prim
