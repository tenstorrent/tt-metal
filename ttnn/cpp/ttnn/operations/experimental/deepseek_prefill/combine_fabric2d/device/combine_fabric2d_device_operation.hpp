// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "combine_fabric2d_types.hpp"
#include "combine_fabric2d_program_factory.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

struct CombineFabric2dDeviceOperation {
    using operation_attributes_t = CombineFabric2dParams;
    using tensor_args_t = CombineFabric2dInputs;
    using spec_return_value_t = ttnn::TensorSpec;
    using tensor_return_value_t = ttnn::Tensor;
    using program_factory_t = std::variant<CombineFabric2dProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d

namespace ttnn::prim {
ttnn::Tensor combine_fabric2d(
    ttnn::MeshDevice* device,
    uint32_t num_links,
    uint32_t num_tokens,
    uint32_t chunk_size_bytes,
    uint32_t num_slots,
    uint32_t axis,
    uint32_t stall_telemetry,
    uint32_t variant,
    tt::tt_fabric::Topology topology,
    const std::optional<ttnn::Tensor>& input = std::nullopt);
}  // namespace ttnn::prim
