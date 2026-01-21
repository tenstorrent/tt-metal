// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/transformer/sdpa/device/sdpa_program_factory.hpp"
#include "ttnn/operations/transformer/sdpa/device/sdpa_device_operation_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/operations/core/core.hpp"
#include <optional>
#include <variant>
#include <tt-metalium/constants.hpp>

namespace ttnn::prim {

struct SDPAOperation {
    using operation_attributes_t = SDPAParams;
    using tensor_args_t = SDPAInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<SDPAProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t& attrs, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor);
};

Tensor sdpa(
    const Tensor& input_tensor_q,
    const Tensor& input_tensor_k,
    const std::optional<Tensor>& input_tensor_v,
    const std::optional<Tensor>& attn_mask,
    const std::optional<Tensor>& page_table_tensor,
    const std::optional<Tensor>& attention_sink,
    bool is_causal,
    std::optional<float> scale,
    std::optional<uint32_t> sliding_window_size,
    std::optional<int64_t> chunk_start_idx,
    bool use_mla,
    std::optional<uint32_t> head_dim_v,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config,
    ttnn::DeviceComputeKernelConfig compute_kernel_config);

}  // namespace ttnn::prim
