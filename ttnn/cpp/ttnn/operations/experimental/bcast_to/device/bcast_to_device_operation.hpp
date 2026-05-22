// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <variant>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/shape/shape.hpp"

namespace ttnn::operations::experimental::broadcast_to {
enum class SubtileBroadcastType {
    NONE,    // both tensors have equal tile dimensions (H & W)
    SCALAR,  // input is a scalar (H = 1, W = 1)
    ROW,     // input has a single tile row
    COL,     // input has a single tile column
};
SubtileBroadcastType get_subtile_broadcast_type(uint32_t a_h, uint32_t a_w, uint32_t b_h, uint32_t b_w);
struct BcastToOperation {
    struct operation_attributes_t {
        const Shape output_shape;
        const MemoryConfig memory_config;
        SubtileBroadcastType subtile_broadcast_type = SubtileBroadcastType::NONE;
    };

    struct tensor_args_t {
        const Tensor& input;
        const std::optional<Tensor>& output;
    };

    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    struct BcastToTileFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);
    };

    using program_factory_t = std::variant<BcastToTileFactory>;
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};
}  // namespace ttnn::operations::experimental::broadcast_to

namespace ttnn::prim {

ttnn::operations::experimental::broadcast_to::BcastToOperation::tensor_return_value_t bcast_to(
    const Tensor& input,
    const ttnn::Shape& output_shape,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output);

}  // namespace ttnn::prim
