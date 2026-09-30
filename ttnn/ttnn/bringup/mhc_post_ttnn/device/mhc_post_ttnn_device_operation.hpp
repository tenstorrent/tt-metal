// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include <tt_stl/reflection.hpp>

#include "mhc_post_ttnn_device_operation_types.hpp"
#include "mhc_post_ttnn_program_factory.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::bringup::mhc_post_ttnn {

struct MhcPostDeviceOperation {
    using operation_attributes_t = MhcPostParams;
    using tensor_args_t = MhcPostInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<MhcPostProgramFactory>;

    // The op's refusals run once on the host before launch (mhc_post_ttnn.cpp, the Python op's exception types);
    // this is only the device-side floor.
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::bringup::mhc_post_ttnn

namespace ttnn::prim::bringup {
ttnn::Tensor mhc_post_ttnn(
    const ttnn::Tensor& input,
    const ttnn::Tensor& residual,
    const ttnn::Tensor& post,
    const ttnn::Tensor& comb,
    const tt::tt_metal::ComputeConfigDescriptor& compute_config);
}  // namespace ttnn::prim::bringup
