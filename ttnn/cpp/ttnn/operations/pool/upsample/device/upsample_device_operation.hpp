// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include <tt-metalium/global_circular_buffer.hpp>
#include "ttnn/operations/pool/upsample/device/upsample_device_operation_types.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include <utility>
#include <vector>

namespace ttnn::prim {

struct UpsampleBilinearProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const UpsampleParams& operation_attributes, const Tensor& input_tensor, Tensor& output_tensor);
};

struct UpsampleMultiCoreInterleavedProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const UpsampleParams& operation_attributes, const Tensor& input_tensor, Tensor& output_tensor);
};

struct UpsampleMultiCoreShardedProgramFactory {
    // create_program_artifacts() uploads the per-core halo lookup config tensor and parks it in
    // op_owned_tensors (held by the program cache) so its lifetime outlives the cached Program.
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const UpsampleParams& operation_attributes, const Tensor& input_tensor, Tensor& output_tensor);
};

struct UpsampleNearestFloatProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const UpsampleParams& operation_attributes, const Tensor& input_tensor, Tensor& output_tensor);
};

struct UpsampleOperation {
    using operation_attributes_t = UpsampleParams;
    using tensor_args_t = Tensor;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<
        UpsampleBilinearProgramFactory,
        UpsampleMultiCoreInterleavedProgramFactory,
        UpsampleMultiCoreShardedProgramFactory,
        UpsampleNearestFloatProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t& args, const Tensor& input);
    static void validate_on_program_cache_miss(const operation_attributes_t& args, const Tensor& input);
    static spec_return_value_t compute_output_specs(const operation_attributes_t& args, const Tensor& input);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t& args, const Tensor& input);
};

ttnn::Tensor upsample(
    const ttnn::Tensor& input_tensor,
    float scale_factor_h,
    float scale_factor_w,
    const std::string& mode,
    const MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    const std::optional<ttnn::operations::sliding_window::SlidingWindowConfig>& sliding_window_config = std::nullopt);

}  // namespace ttnn::prim
