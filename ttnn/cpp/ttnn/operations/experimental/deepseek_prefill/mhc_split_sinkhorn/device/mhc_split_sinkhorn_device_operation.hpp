// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "ttnn/tensor/tensor.hpp"
#include "mhc_split_sinkhorn_program_factory.hpp"
#include "mhc_split_sinkhorn_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct MhcSplitSinkhornDeviceOperation {
    using operation_attributes_t = MhcSplitSinkhornParams;
    using tensor_args_t = MhcSplitSinkhornTensorArgs;
    using spec_return_value_t = std::array<tt::tt_metal::TensorSpec, 3>;
    using tensor_return_value_t = std::array<Tensor, 3>;
    using program_factory_t = std::variant<MhcSplitSinkhornProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
std::array<Tensor, 3> mhc_split_sinkhorn(
    const Tensor& mixes, const Tensor& consts, uint32_t n, uint32_t sinkhorn_iters, float eps);
}  // namespace ttnn::prim
