// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>
#include <variant>

#include <tt_stl/reflection.hpp>

#include "mhc_pre_ttnn_device_operation_types.hpp"
#include "mhc_pre_ttnn_program_factory.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::bringup::mhc_pre_ttnn {

struct MhcPreDeviceOperation {
    using operation_attributes_t = MhcPreParams;
    using tensor_args_t = MhcPreInputs;
    using spec_return_value_t =
        std::tuple<tt::tt_metal::TensorSpec, tt::tt_metal::TensorSpec, tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::tuple<Tensor, Tensor, Tensor>;  // (y, post, comb)
    using program_factory_t = std::variant<MhcPreProgramFactory>;

    // The op's refusals run once on the host before launch (mhc_pre_ttnn.cpp, the Python op's exception types);
    // this is only the device-side floor.
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::bringup::mhc_pre_ttnn

namespace ttnn::prim::bringup {
std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> mhc_pre_ttnn(
    const ttnn::Tensor& input,
    const ttnn::Tensor& proj_weight,
    const ttnn::Tensor& proj_bias,
    const ttnn::operations::bringup::mhc_pre_ttnn::MhcPreParams& params);
}  // namespace ttnn::prim::bringup
