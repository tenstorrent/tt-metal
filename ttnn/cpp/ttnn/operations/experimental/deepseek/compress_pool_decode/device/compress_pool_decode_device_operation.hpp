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
    // Index key k_norm(latent @ index_wk), RoPE'd, into idx_key_cache row ``entry``.
    bool has_index_key = false;
    bool index_norm_weight_is_tile = true;
    uint32_t index_head_dim = 0;
    // The RoPE'd row into comp_kv at ``entry`` / into dense_kv at ``dense_entry``.
    bool has_comp_kv = false;
    bool has_dense = false;
    bool has_page_row = false;
    bool has_key_row = false;
    // Rows per page-table block of idx_key_cache / comp_kv.
    uint32_t block_rows = 0;
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
    std::optional<Tensor> index_wk;
    std::optional<Tensor> index_norm_weight;
    std::optional<Tensor> idx_key_cache;
    std::optional<Tensor> comp_kv;
    std::optional<Tensor> page_table;
    std::optional<Tensor> entry;
    std::optional<Tensor> page_row;
    std::optional<Tensor> dense_kv;
    std::optional<Tensor> dense_entry;
    std::optional<Tensor> key_row;
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
    bool do_pool,
    const std::optional<ttnn::Tensor>& index_wk,
    const std::optional<ttnn::Tensor>& index_norm_weight,
    const std::optional<ttnn::Tensor>& idx_key_cache,
    const std::optional<ttnn::Tensor>& comp_kv,
    const std::optional<ttnn::Tensor>& page_table,
    const std::optional<ttnn::Tensor>& entry,
    const std::optional<ttnn::Tensor>& page_row,
    const std::optional<ttnn::Tensor>& dense_kv,
    const std::optional<ttnn::Tensor>& dense_entry,
    const std::optional<ttnn::Tensor>& key_row);

}  // namespace ttnn::prim
