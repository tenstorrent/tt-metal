// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>
#include <variant>
#include <tt-metalium/sub_device_types.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "high_bw_all_gather_device_operation_types.hpp"
#include "high_bw_all_gather_unicast_factory.hpp"

namespace ttnn::operations::experimental::high_bw_all_gather {

struct HighBwAllGatherDeviceOperation {
    using operation_attributes_t = HighBwAllGatherParams;
    using tensor_args_t = HighBwAllGatherInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using topology_return_value_t = std::vector<tt::tt_metal::TensorTopology>;
    using program_factory_t = std::variant<HighBwAllGatherUnicastFactory>;

    // The selected batch slot and valid prefix are patched into kernel runtime arguments. Hash only
    // their presence, so a serving loop reuses one compiled program as either value changes.
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static topology_return_value_t compute_output_topologies(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::high_bw_all_gather

namespace ttnn::prim {

Tensor high_bw_all_gather(
    const Tensor& input_tensor,
    const ttnn::Tensor& output_tensor,
    int32_t dim,
    std::optional<uint32_t> cluster_axis,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grid = std::nullopt,
    std::optional<uint32_t> num_links = std::nullopt,
    std::optional<uint32_t> input_batch_index = std::nullopt,
    std::optional<uint32_t> gathered_dim_size = std::nullopt,
    const std::optional<Tensor>& page_bundle_indices = std::nullopt,
    uint32_t kv_cache_page_size = 32,
    uint32_t kv_cache_num_layers = 1,
    uint32_t kv_cache_layer_idx = 0);

}  // namespace ttnn::prim
