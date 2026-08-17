// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "topk_large_indices_device_operation_types.hpp"
#include "topk_large_indices_program_factory.hpp"

namespace ttnn::operations::experimental::topk_large_indices {

struct TopkLargeIndicesDeviceOperation {
    using operation_attributes_t = topk_large_indices::operation_attributes_t;
    using tensor_args_t = topk_large_indices::tensor_args_t;
    using tensor_return_value_t = topk_large_indices::tensor_return_value_t;
    using spec_return_value_t = topk_large_indices::spec_return_value_t;

    using program_factory_t =
        std::variant<program::TopkLargeIndicesProgramFactory, program::TopkLargeIndicesMultiCoreProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(const operation_attributes_t& attrs, const tensor_args_t& tensor_args);
    static void validate_on_program_cache_hit(const operation_attributes_t& attrs, const tensor_args_t& tensor_args);
    static ttsl::hash::hash_t compute_program_hash(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor,
        uint32_t k,
        std::optional<uint32_t> valid_length,
        bool return_values,
        std::optional<uint32_t> num_slices,
        bool tile_output,
        std::optional<DataType> index_dtype);
};

}  // namespace ttnn::operations::experimental::topk_large_indices

namespace ttnn::experimental {

// Input may be ROW_MAJOR or TILE layout (BFLOAT16, interleaved). With the defaults the outputs
// are ROW_MAJOR and the indices are UINT32 — bit-identical behavior to before the opt-ins below
// existed. tile_output=true emits TILE-layout outputs directly (k must be a multiple of 32;
// tile padding rows are zero-filled). index_dtype=UINT16 narrows the indices output (searched
// width must be <= 65535; the -inf sentinel becomes 0xFFFF).
Tensor topk_large_indices(
    const Tensor& input_tensor,
    uint32_t k,
    std::optional<uint32_t> valid_length = std::nullopt,
    std::optional<uint32_t> num_slices = std::nullopt,
    bool tile_output = false,
    std::optional<DataType> index_dtype = std::nullopt);

// (values, indices): values are BFLOAT16, sorted descending to match the indices;
// sentinel-index (-inf) lanes carry exact bf16 -inf values. Layout/dtype opt-ins as above.
std::tuple<Tensor, Tensor> topk_large_indices_with_values(
    const Tensor& input_tensor,
    uint32_t k,
    std::optional<uint32_t> valid_length = std::nullopt,
    std::optional<uint32_t> num_slices = std::nullopt,
    bool tile_output = false,
    std::optional<DataType> index_dtype = std::nullopt);

}  // namespace ttnn::experimental
