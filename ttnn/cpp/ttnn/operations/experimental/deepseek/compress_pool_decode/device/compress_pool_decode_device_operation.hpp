// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include "compress_pool_decode_program_factory.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek::compress_pool_decode {

struct CompressPoolDecodeParams {
    float epsilon = 1e-6f;
    bool do_pool = true;
    bool has_window = false;
    bool has_prev_slot = false;
    bool has_rope = false;
    uint32_t head_dim = 0;
    uint32_t rope_dim = 0;
    bool kv_sharded = false;
    bool gate_sharded = false;
    bool norm_weight_is_tile = true;
};

struct CompressPoolDecodeInputs {
    const Tensor& kv;
    const Tensor& norm_weight;
    std::optional<Tensor> gate;
    std::optional<Tensor> win_kv;
    std::optional<Tensor> win_gate;
    std::optional<Tensor> win_slot;
    std::optional<Tensor> prev_slot;
    std::optional<Tensor> cos;
    std::optional<Tensor> sin;
};

struct CompressPoolDecodeDeviceOperation {
    using operation_attributes_t = CompressPoolDecodeParams;
    using tensor_args_t = CompressPoolDecodeInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<CompressPoolDecodeProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deepseek::compress_pool_decode

namespace ttnn::prim {

std::vector<ttnn::Tensor> compress_pool_decode(
    const ttnn::Tensor& kv,
    const ttnn::Tensor& norm_weight,
    float epsilon,
    const std::optional<ttnn::Tensor>& gate,
    const std::optional<ttnn::Tensor>& win_kv,
    const std::optional<ttnn::Tensor>& win_gate,
    const std::optional<ttnn::Tensor>& win_slot,
    const std::optional<ttnn::Tensor>& prev_slot,
    const std::optional<ttnn::Tensor>& cos,
    const std::optional<ttnn::Tensor>& sin,
    bool do_pool);

}  // namespace ttnn::prim
