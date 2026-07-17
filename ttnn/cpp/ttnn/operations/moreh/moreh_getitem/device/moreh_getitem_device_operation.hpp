// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/types.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::moreh::moreh_getitem {
struct MorehGetItemOperation {
    struct operation_attributes_t {
        const ttsl::SmallVector<uint32_t> index_dims;
        // const CoreRange core_range;
        const MemoryConfig memory_config;
    };

    struct tensor_args_t {
        const Tensor& input;
        const std::vector<Tensor>& index_tensors;
        const std::optional<Tensor>& output;
    };

    using spec_return_value_t = ttnn::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct MorehGetItemRmFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output_tensor);
    };

    struct MorehGetItemTilizedFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output_tensor);
    };

    using program_factory_t = std::variant<MorehGetItemRmFactory, MorehGetItemTilizedFactory>;

    static void validate_inputs(const operation_attributes_t&, const tensor_args_t&);
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::moreh::moreh_getitem

namespace ttnn::prim {
ttnn::operations::moreh::moreh_getitem::MorehGetItemOperation::tensor_return_value_t moreh_getitem(
    const Tensor& input,
    const std::vector<Tensor>& index_tensors,
    const ttsl::SmallVector<uint32_t>& index_dims,
    const std::optional<Tensor>& output,
    const std::optional<MemoryConfig>& memory_config);
}
