// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "mhc_post_mix_program_factory.hpp"
#include "mhc_post_mix_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct MhcPostMixDeviceOperation {
    using operation_attributes_t = MhcPostMixParams;
    using tensor_args_t = MhcPostMixTensorArgs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<MhcPostMixProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
Tensor mhc_post_mix(
    const Tensor& x, const Tensor& y, const Tensor& post, const Tensor& comb, uint32_t n, bool collapse);
}  // namespace ttnn::prim
