// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "lnbw_program_factory.hpp"
#include "lnbw_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct LnBwDeviceOperation {
    using operation_attributes_t = LnBwParams;
    using tensor_args_t = LnBwInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<LnBwProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
Tensor fused_ln_bw(
    const Tensor& gy,
    const Tensor& x,
    const Tensor& red,
    const Tensor& n,
    const Tensor& gamma,
    uint32_t W,
    uint32_t eps_bits);
}  // namespace ttnn::prim
