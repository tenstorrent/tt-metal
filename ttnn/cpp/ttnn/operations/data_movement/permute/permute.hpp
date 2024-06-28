// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core.hpp"
#include "ttnn/validation.hpp"
#include "tt_eager/tt_dnn/op_library/permute/permute_op.hpp"
#include "tt_eager/tt_dnn/op_library/run_operation.hpp"

#include "ttnn/cpp/ttnn/types.hpp"

inline bool is_on_device(const Tensor& t) {
    return t.storage_type() == tt::tt_metal::StorageType::DEVICE or
           t.storage_type() == tt::tt_metal::StorageType::MULTI_DEVICE;
}

inline bool has_tile_padding(const Tensor& t) {
    if (t.get_shape().rank() > 1) {
        auto the_shape = t.get_shape();
        auto the_shape_with_padding = t.get_shape().with_tile_padding();
        return the_shape[-1] != the_shape_with_padding[-1] or the_shape[-2] != the_shape_with_padding[-2];
    }
    return false;
}


namespace ttnn {
namespace operations::data_movement {

constexpr uint8_t DefaultQueueId = 0;

struct Permute {
    static inline const std::array<TensorSchema, 1> input_tensor_schemas() {
        return {ttnn::TensorSchema{
            2,  // min rank
            4,  // max rank
            {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::uint16, ttnn::int32, ttnn::uint32, ttnn::float32},
            {ttnn::TILE_LAYOUT},
            true,   // can_be_on_device
            true,  // can_be_on_cpu
            false,  // can_be_scalar
            false   // is_optional}
        }};
    }

    template <typename... Args>
    static auto input_tensors_to_validate(uint8_t queue_id, const Tensor& input_tensor, Args&&... args) {
        return std::forward_as_tuple(input_tensor);
    }

    static inline ttnn::Tensor execute_on_worker_thread(
        uint8_t queue_id,
        const ttnn::Tensor &input_tensor,
        const std::vector<int>& dims,
        const std::optional<MemoryConfig>& memory_config) {

        const bool initial_input_tensor_on_device = is_on_device(input_tensor);
        const auto input_layout = input_tensor.get_layout();
        const auto input_rank = input_tensor.get_shape().rank();

        TT_FATAL(input_rank <= 4);
        TT_FATAL(
            input_rank == dims.size(),
            "The number of dimensions in the tensor input does not match the length of the desired ordering");

        auto adjust_order = [](const std::vector<int>& dims) {
            std::vector<std::int64_t> new_order;
            TT_FATAL(dims.size() <= 4);
            int additional_ranks = 4 - dims.size();
            for (int i = 0; i < additional_ranks; i++) {
                new_order.push_back(i);
            }
            for (int i = 0; i < dims.size(); i++) {
                new_order.push_back(dims.at(i) + additional_ranks);
            }
            return new_order;
        };
        auto itensor = (input_tensor.get_shape().rank() < 4) ? ttnn::unsqueeze_to_4D(input_tensor) : input_tensor;
        auto iorder = adjust_order(dims);

        if (has_tile_padding(itensor)) {
            itensor = ttnn::to_layout(itensor, ttnn::ROW_MAJOR_LAYOUT, std::nullopt, std::nullopt, (Device*)nullptr);
        }

        TT_FATAL(is_on_device(itensor) and itensor.get_shape().rank() == 4);
        auto output_tensor = tt::tt_metal::permute(itensor, iorder, memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG));
        output_tensor = ttnn::to_layout(output_tensor, input_layout, std::nullopt, std::nullopt, (Device*)nullptr);

        if (input_rank < 4) {
            const auto shape = output_tensor.get_shape();
            const auto full_shape = output_tensor.get_shape().with_tile_padding();
            std::vector<uint32_t> shape_vec{};
            std::vector<uint32_t> full_shape_vec{};
            int i = 0;
            while (i < 3 and shape[i] == 1) i++;
            for (; i < shape.rank(); i++) {
                shape_vec.push_back(shape[i]);
                full_shape_vec.push_back(full_shape[i]);
            }
            auto metal_shape = tt::tt_metal::Shape(shape_vec, full_shape_vec);
            output_tensor = ttnn::reshape(output_tensor, ttnn::Shape(metal_shape));
        }

        if (initial_input_tensor_on_device and not is_on_device(output_tensor)) {
            output_tensor = ttnn::to_device(output_tensor, input_tensor.device(), memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG));
        }

        return output_tensor;
    }

    template <typename... Args>
    static auto input_tensors_to_validate(const Tensor& input_tensor, Args&&... args) {
        return std::forward_as_tuple(input_tensor);
    }

    static inline auto execute_on_worker_thread(
        const ttnn::Tensor &input_tensor,
        const std::vector<int>& dims,
        const std::optional<MemoryConfig>& memory_config) {
        return execute_on_worker_thread(DefaultQueueId, input_tensor, dims, memory_config);
    }
};

}  // namespace operations::data_movement

constexpr auto permute = ttnn::register_operation<ttnn::operations::data_movement::Permute>("ttnn::permute");

}  // namespace ttnn
