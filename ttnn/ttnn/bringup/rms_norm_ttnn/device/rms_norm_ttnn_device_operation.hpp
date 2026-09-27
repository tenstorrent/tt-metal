// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include <tt_stl/reflection.hpp>

#include "rms_norm_ttnn_device_operation_types.hpp"
#include "rms_norm_ttnn_program_factory.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::bringup::rms_norm_ttnn {

struct RmsNormDeviceOperation {
    using operation_attributes_t = RmsNormParams;
    using tensor_args_t = RmsNormInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<RmsNormProgramFactory>;

    // The op's refusals run once, on the host, before this op is launched (the host-side checks in
    // rms_norm_ttnn.cpp, which keep the Python op's exception types); this is only the device-side floor.
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::bringup::rms_norm_ttnn

namespace ttnn::prim::bringup {
ttnn::Tensor rms_norm_ttnn(
    const ttnn::Tensor& input,
    const std::optional<ttnn::Tensor>& weight,
    const std::optional<ttnn::Tensor>& bias,
    const std::optional<ttnn::Tensor>& residual,
    double epsilon,
    const tt::tt_metal::ComputeConfigDescriptor& compute_config,
    uint32_t subblock_w,
    bool inplace,
    const tt::tt_metal::MemoryConfig& output_mem_config);
}  // namespace ttnn::prim::bringup
