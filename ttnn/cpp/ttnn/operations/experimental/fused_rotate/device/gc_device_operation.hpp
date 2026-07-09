// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "gc_program_factory.hpp"
#include "gc_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct FusedGcDeviceOperation {
    using operation_attributes_t = FusedGcParams;
    using tensor_args_t = FusedGcInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<FusedGcProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
Tensor fused_rotate_gc(
    const Tensor& gout,
    const Tensor& xin,
    const Tensor& sel,
    uint32_t n_out,
    uint32_t n_in,
    uint32_t W,
    const std::vector<uint32_t>& is_,
    const std::vector<uint32_t>& js);
}  // namespace ttnn::prim
