// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"

#include "ttnn/device_operation.hpp"
#include <tt-metalium/global_circular_buffer.hpp>
#include "ttnn/operations/pool/upsample/device/upsample_device_operation_types.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/operations/pool/upsample/device/upsample_bilinear_program_factory_multicore.hpp"
#include "ttnn/operations/pool/upsample/device/upsample_program_factory_multicore_interleaved.hpp"
#include "ttnn/operations/pool/upsample/device/upsample_program_factory_multicore_sharded.hpp"

namespace ttnn::operations::pool::upsample {

struct UpsampleOperation {
    using operation_attributes_t = UpsampleParams;
    using tensor_args_t = UpsampleInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<
        program::UpsampleBilinearProgramFactory,
        program::UpsampleMultiCoreInterleavedProgramFactory,
        program::UpsampleMultiCoreShardedProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& args, const tensor_args_t& tensor_args);
    static void validate_on_program_cache_miss(const operation_attributes_t& args, const tensor_args_t& tensor_args);
    static void validate_on_program_cache_hit(const operation_attributes_t& args, const tensor_args_t& tensor_args);
    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& args, const tensor_args_t& tensor_args);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& args, const tensor_args_t& tensor_args);
};

}  // namespace ttnn::operations::pool::upsample

namespace ttnn::prim {
ttnn::Tensor upsample(
    const ttnn::Tensor& input_tensor,
    int scale_factor_h,
    int scale_factor_w,
    const std::string& mode,
    const MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    const std::optional<ttnn::operations::sliding_window::SlidingWindowConfig>& sliding_window_config = std::nullopt);
}  // namespace ttnn::prim
