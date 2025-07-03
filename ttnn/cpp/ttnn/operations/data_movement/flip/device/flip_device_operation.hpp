// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <optional>

#include <tt_stl/span.hpp>

#include "ttnn/core.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

struct FlipDeviceOperation {
    struct operation_attributes_t {
        const SmallVector<uint32_t> dims;
        const MemoryConfig output_mem_config;
    };

    // Implementation for a row major tensor where the row dimension is not moved in the permutation
    struct MultiCoreRowInvariant {};

    // Implementation for a row major tensor where the row dimension is moved in the permutation
    struct MultiCoreBlockedGeneric {};

    // Implementation for when the tile is not broken apart
    struct MultiCoreTileInvariant {};

    // Implemention for when only one of the height dimension (rank - 2) and the width dimension is swapped with another
    // dimension
    struct MultiCoreTileRowInvariant {};

    // Implementation for when both the height and width dimension is swapped around in the permutation
    struct MultiCoreTiledGeneric {};

    using program_factory_t = std::variant<
        MultiCoreRowInvariant,
        MultiCoreBlockedGeneric,
        MultiCoreTileInvariant,
        MultiCoreTileRowInvariant,
        MultiCoreTiledGeneric>;

    // Mandatory methods

    // Select the program factory based on the operation attributes and tensor args
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    // Validate the operation when it creates a program.
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    // Empty as there doesn't seem to be any complicated hashing requirement
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    // Compute the output shapes based on the operation attributes and tensor args
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    // Create the output tensors based on the operation attributes and tensor args
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    // API call to map user arguments to operation attributes and tensor args.
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor,
        const SmallVector<uint32_t>& dims,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<Tensor> optional_output_tensor);
};

}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {
// Register the operation with the ttnn::register_operation API to make it available to the user as ttnn::prim::example
constexpr auto permute =
    ttnn::register_operation<"ttnn::prim::flip", ttnn::operations::data_movement::FlipDeviceOperation>();
}  // namespace ttnn::prim
