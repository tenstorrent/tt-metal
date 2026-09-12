// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <optional>
#include <variant>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct CsaRuntimeParams {
    uint32_t seq_len_actual;
    uint32_t first_token_position;
    uint32_t cluster_axis;
};

struct CsaStateInputs {
    const Tensor& kv;
    const Tensor& gate;
    const Tensor& position_bias;
    const Tensor& base_kv_state;
    const Tensor& base_score_state;
};

struct CsaStatePreparationProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const CsaRuntimeParams&,
        const CsaStateInputs&,
        std::array<Tensor, 2>&,
        const std::optional<ttnn::MeshCoordinate>&);
};

struct CsaStatePreparationDeviceOperation {
    using operation_attributes_t = CsaRuntimeParams;
    using tensor_args_t = CsaStateInputs;
    using spec_return_value_t = std::array<tt::tt_metal::TensorSpec, 2>;
    using topology_return_value_t = std::vector<tt::tt_metal::TensorTopology>;
    using tensor_return_value_t = std::array<Tensor, 2>;
    using program_factory_t = std::variant<CsaStatePreparationProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static topology_return_value_t compute_output_topologies(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

struct CsaCompressionInputs {
    const Tensor& kv;
    const Tensor& gate;
    const Tensor& position_bias;
    const Tensor& predecessor_kv_state;
    const Tensor& predecessor_score_state;
};

struct CsaCompressionProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const CsaRuntimeParams&,
        const CsaCompressionInputs&,
        std::array<Tensor, 3>&,
        const std::optional<ttnn::MeshCoordinate>&);
};

struct CsaCompressionDeviceOperation {
    using operation_attributes_t = CsaRuntimeParams;
    using tensor_args_t = CsaCompressionInputs;
    using spec_return_value_t = std::array<tt::tt_metal::TensorSpec, 3>;
    using topology_return_value_t = std::vector<tt::tt_metal::TensorTopology>;
    using tensor_return_value_t = std::array<Tensor, 3>;
    using program_factory_t = std::variant<CsaCompressionProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static topology_return_value_t compute_output_topologies(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
std::array<Tensor, 2> csa_prepare_state(
    const Tensor& kv,
    const Tensor& gate,
    const Tensor& position_bias,
    const Tensor& base_kv_state,
    const Tensor& base_score_state,
    uint32_t seq_len_actual,
    uint32_t first_token_position,
    uint32_t cluster_axis);

std::array<Tensor, 3> csa_compress(
    const Tensor& kv,
    const Tensor& gate,
    const Tensor& position_bias,
    const Tensor& predecessor_kv_state,
    const Tensor& predecessor_score_state,
    uint32_t seq_len_actual,
    uint32_t first_token_position,
    uint32_t cluster_axis);
}  // namespace ttnn::prim
