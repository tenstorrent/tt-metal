// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "all_reduce_create_qkv_heads_device_operation_types.hpp"
#include "all_reduce_create_qkv_heads_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

#include <functional>
#include <optional>
#include <variant>

namespace ttnn::experimental::prim {

struct AllReduceCreateQkvHeadsDeviceOperation {
    using operation_attributes_t = AllReduceCreateQkvHeadsParams;
    using tensor_args_t = AllReduceCreateQkvHeadsInputs;
    using spec_return_value_t = AllReduceCreateQkvHeadsResultSpec;
    using tensor_return_value_t = AllReduceCreateQkvHeadsResult;
    using program_factory_t = std::variant<AllReduceCreateQkvHeadsMeshWorkloadFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
ttnn::experimental::prim::AllReduceCreateQkvHeadsResult all_reduce_create_qkv_heads(
    const Tensor& input_tensor,
    Tensor& buffer_tensor,
    const Tensor& batch_offset_tensor,
    uint32_t num_links,
    uint32_t ring_size,
    const MemoryConfig& all_reduce_mem_config,
    ttnn::ccl::Topology topology,
    const GlobalSemaphore& semaphore,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    uint32_t head_dim,
    bool use_noc1_only,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    bool input_on_subcoregrids,
    std::optional<uint32_t> slice_size,
    const MemoryConfig& final_mem_config,
    DataType dtype,
    uint32_t cluster_axis);
}  // namespace ttnn::prim
