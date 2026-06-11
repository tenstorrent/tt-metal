// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "indexer_score_device_operation_types.hpp"
#include "indexer_score_program_factory.hpp"

namespace ttnn::operations::experimental::deepseek::indexer {

struct IndexerScoreDeviceOperation {
    using operation_attributes_t = indexer::operation_attributes_t;
    using tensor_args_t = indexer::tensor_args_t;
    using spec_return_value_t = indexer::spec_return_value_t;
    using tensor_return_value_t = indexer::tensor_return_value_t;
    using program_factory_t = std::variant<program::IndexerScoreProgramFactory>;

    // No custom compute_program_hash: operation_attributes_t (incl. is_causal, chunk_start_idx,
    // and every IndexerScoreProgramConfig field) is a reflectable aggregate, so the default
    // reflection hash already keys distinct programs on all of them. Do not add a hand-rolled
    // hash that drops fields.
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& q,
        const Tensor& k,
        const Tensor& weights,
        bool is_causal,
        uint32_t chunk_start_idx,
        const IndexerScoreProgramConfig& program_config);
};

}  // namespace ttnn::operations::experimental::deepseek::indexer

namespace ttnn::prim {

ttnn::Tensor indexer_score(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& weights,
    bool is_causal,
    uint32_t chunk_start_idx,
    const ttnn::operations::experimental::deepseek::indexer::IndexerScoreProgramConfig& program_config);

}  // namespace ttnn::prim
