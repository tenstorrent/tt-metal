// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "attn_res_accum_stats_device_operation_types.hpp"
#include "attn_res_accum_stats_program_factory.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::prim {

struct AttnResAccumStatsDeviceOperation {
    using operation_attributes_t = AttnResAccumStatsParams;
    using tensor_args_t = AttnResAccumStatsInputs;
    using spec_return_value_t = std::array<tt::tt_metal::TensorSpec, 2>;
    using tensor_return_value_t = std::array<Tensor, 2>;
    using program_factory_t = std::variant<AttnResAccumStatsProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::array<Tensor, 2> attn_res_accum_stats(
    const Tensor& a,
    const Tensor& b,
    const Tensor& q,
    DataType stats_dtype,
    const MemoryConfig& total_mem_config,
    const MemoryConfig& stats_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config);

}  // namespace ttnn::prim
