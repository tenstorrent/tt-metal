// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <variant>
#include <vector>

#include "ttnn/decorators.hpp"
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
        struct shared_variables_t {
            tt::tt_metal::KernelHandle reader_kernel_id;
            tt::tt_metal::KernelHandle writer_kernel_id;
            tt::tt_metal::KernelHandle compute_kernel_id;
            CoreCoord compute_with_storage_grid_size;
        };

        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);
    };

    using program_factory_t = std::variant<BcastToTileFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input,
        const ttnn::Shape& output_shape,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<Tensor>& output);
};
}  // namespace ttnn::operations::experimental::broadcast_to

namespace ttnn::prim {
constexpr auto bcast_to =
    ttnn::register_operation<"ttnn::prim::bcast_to", ttnn::operations::experimental::broadcast_to::BcastToOperation>();
}
