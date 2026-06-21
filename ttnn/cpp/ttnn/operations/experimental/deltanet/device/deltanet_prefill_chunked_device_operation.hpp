// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <functional>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "deltanet_prefill_chunked_program_factory.hpp"
#include "deltanet_prefill_chunked_device_operation_types.hpp"

namespace ttnn::operations::experimental::deltanet {

struct DeltaNetPrefillChunkedDeviceOperation {
    using operation_attributes_t = DeltaNetPrefillChunkedParams;
    using tensor_args_t = DeltaNetPrefillChunkedInputs;
    using spec_return_value_t = std::vector<ttnn::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<DeltaNetPrefillChunkedProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deltanet

namespace ttnn::prim {

std::vector<Tensor> deltanet_prefill_chunked(
    const Tensor& k, const Tensor& q, const Tensor& v, const Tensor& z,
    const Tensor& Kdec, const Tensor& KiT, const Tensor& Qd,
    const Tensor& dcol, const Tensor& betacol, const Tensor& dlast,
    const Tensor& recurrent_state, const Tensor& norm_weight,
    uint32_t num_heads, uint32_t k_head_dim, uint32_t v_head_dim,
    uint32_t chunk, uint32_t n_chunks, uint32_t seq_len,
    const std::optional<tt::tt_metal::MemoryConfig>& output_memory_config = std::nullopt);

}  // namespace ttnn::prim
