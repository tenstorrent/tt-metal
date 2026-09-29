// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "gdn_gates_device_operation_types.hpp"
#include "gdn_gates_program_factory.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::experimental::prim {

struct GdnGatesOperation {
    using operation_attributes_t = GdnGatesParams;
    using tensor_args_t = GdnGatesInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<GdnGatesProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

std::tuple<Tensor, Tensor> gdn_gates(
    const Tensor& gab,
    const Tensor& dt_bias,
    const Tensor& a_neg,
    uint32_t a_col_offset,
    uint32_t b_col_offset,
    uint32_t num_heads,
    float beta_scale,
    const tt::tt_metal::MemoryConfig& output_mem_config);

}  // namespace ttnn::experimental::prim
