// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/types.hpp"
#include "pair_pool_device_operation_types.hpp"
#include "pair_pool_program_factory.hpp"

namespace ttnn::operations::pair_pool {

struct PairPoolDeviceOperation {
    using operation_attributes_t = PairPoolParams;
    using tensor_args_t = PairPoolInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<PairPoolProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    // The public entry runs check_support on every call, before the launch: a hit checks nothing again.
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    // The attributes and the input's spec; the caller's output_tensor, present or absent, is left out.
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

// The Python asserts, in order and with their messages: throws PairPoolAssertion.
void check_support(const PairPoolParams& attrs, const PairPoolInputs& t);

}  // namespace ttnn::operations::pair_pool

namespace ttnn::prim {

Tensor pair_pool(
    const Tensor& input,
    const std::optional<Tensor>& output,
    const ttnn::operations::pair_pool::PairPoolParams& params);

}  // namespace ttnn::prim
