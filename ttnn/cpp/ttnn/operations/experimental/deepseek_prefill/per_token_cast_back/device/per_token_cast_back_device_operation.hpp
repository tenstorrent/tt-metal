// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "per_token_cast_back_device_operation_types.hpp"
#include "per_token_cast_back_program_factory.hpp"
#include "per_token_cast_back_masked_program_factory.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim::per_token_cast_back {

struct PerTokenCastBackDeviceOperation {
    using operation_attributes_t = PerTokenCastBackParams;
    using tensor_args_t = PerTokenCastBackInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    // Two coexisting program-factory contracts: the original single-program factory (Contract 1)
    // for the plain path, and a per-mesh-coordinate workload-descriptor factory (Contract 2) for the
    // masked path (which needs linearized_mesh_coord to compute the per-device expert window).
    using program_factory_t = std::variant<PerTokenCastBackProgramFactory, MaskedPerTokenCastBackProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim::per_token_cast_back

namespace ttnn::prim {
ttnn::Tensor per_token_cast_back(
    const Tensor& input_e4m3,
    const Tensor& input_scale,
    tt::tt_metal::DataType output_dtype,
    const tt::tt_metal::MemoryConfig& output_memory_config,
    const std::optional<Tensor>& expert_token_counts = std::nullopt,
    const std::optional<Tensor>& expert_region_offsets = std::nullopt,
    const std::optional<Tensor>& metadata = std::nullopt,
    uint32_t experts_per_chip = 0,
    uint32_t dispatch_group_size = 0);
}  // namespace ttnn::prim
