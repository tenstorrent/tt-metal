// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "shared_expert_ffn_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::shared_expert_ffn {

// The unified routed-expert program in its dense mode: one expert owning every row of x, so no
// counts / index / region-offset vectors exist on device.
struct SharedExpertFfnProgramFactory {
    // Built per mesh coordinate: the coordinate picks this device's emb slice of the output.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const SharedExpertFfnParams& operation_attributes,
        const SharedExpertFfnInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& coord);

    // The unified program binds x again in place of the dense mode's unread aux buffers, so x
    // appears more than once and resolve_bindings bails. Patch the common args directly.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const SharedExpertFfnParams& operation_attributes,
        const SharedExpertFfnInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& coord = std::nullopt);
};

struct SharedExpertFfnDeviceOperation {
    using operation_attributes_t = SharedExpertFfnParams;
    using tensor_args_t = SharedExpertFfnInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = ttnn::Tensor;
    using program_factory_t = std::variant<SharedExpertFfnProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::shared_expert_ffn

namespace ttnn::prim {

ttnn::Tensor shared_expert_ffn(
    const ttnn::Tensor& x,
    const ttnn::Tensor& gate_proj,
    const ttnn::Tensor& up_proj,
    const ttnn::Tensor& down_proj,
    ttnn::operations::experimental::deepseek_prefill::shared_expert_ffn::RoutedExpertActivation activation,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    const CoreCoord& core_grid_origin,
    std::optional<uint32_t> cluster_axis);

}  // namespace ttnn::prim
