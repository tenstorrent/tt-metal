// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "fused_rotate_program_factory.hpp"
#include "fused_rotate_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct FusedRotateDeviceOperation {
    using operation_attributes_t = FusedRotateParams;
    using tensor_args_t = FusedRotateInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<FusedRotateProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
Tensor fused_rotate(
    const Tensor& x_flat,
    const Tensor& coef_exp,
    uint32_t n_in,
    uint32_t n_out,
    uint32_t W,
    const std::vector<uint32_t>& deg,
    const std::vector<uint32_t>& ks,
    const std::vector<uint32_t>& js);
}  // namespace ttnn::prim
