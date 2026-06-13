// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <tuple>
#include <variant>
#include <vector>

#include "nemotron3_mamba2_decode_owned_device_operation_types.hpp"
#include "nemotron3_mamba2_decode_owned_program_factory.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct Nemotron3Mamba2DecodeOwnedDeviceOperation {
    using operation_attributes_t = Nemotron3Mamba2DecodeOwnedParams;
    using tensor_args_t = Nemotron3Mamba2DecodeOwnedInputs;
    using spec_return_value_t = std::tuple<TensorSpec, TensorSpec>;
    using tensor_return_value_t = std::tuple<Tensor, Tensor>;
    using program_factory_t = std::variant<Nemotron3Mamba2DecodeOwnedProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t& args, const tensor_args_t& tensor_args);
    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& args, const tensor_args_t& tensor_args);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& args, const tensor_args_t& tensor_args);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

// Mamba2 SSD decode step. Returns (ssm_state_out, y).
// See research/mm7_g1_mamba2_kernel_design.md §2 for the contract.
std::tuple<Tensor, Tensor> nemotron3_mamba2_decode_owned(
    const Tensor& x,
    const Tensor& z,
    const Tensor& dt,
    const Tensor& dt_bias,
    const Tensor& A_log,
    const Tensor& D,
    const Tensor& B_in,
    const Tensor& C_in,
    const Tensor& ssm_state,
    bool debug_fill = false,
    uint32_t debug_mode = 0,
    const std::optional<MemoryConfig>& output_memory_config = std::nullopt,
    const std::optional<Tensor>& preallocated_y = std::nullopt);

}  // namespace ttnn::prim
