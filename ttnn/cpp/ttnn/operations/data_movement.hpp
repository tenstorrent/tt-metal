// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_eager/tensor/types.hpp"
#include "tt_eager/tt_dnn/op_library/concat/concat_op.hpp"
#include "tt_eager/tt_dnn/op_library/pad/pad_op.hpp"
#include "tt_eager/tt_dnn/op_library/permute/permute_op.hpp"
#include "tt_eager/tt_dnn/op_library/repeat/repeat_op.hpp"
#include "tt_eager/tt_dnn/op_library/composite/composite_ops.hpp"
#include "tt_eager/tt_dnn/op_library/upsample/upsample_op.hpp"
#include "ttnn/cpp/ttnn/operations/core.hpp"

#include <ranges>

namespace ttnn {
namespace operations {
namespace data_movement {

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

inline bool has_tile_padding(const Tensor& t, int dim) {
    int rank = t.get_shape().rank();
    // Wrap dim
    dim = dim < 0 ? rank + dim : dim;
    TT_FATAL(
        dim >= 0 and dim < rank,
        "ttnn: Dimension out of range: dim {} cannot be used for tensors of rank {}",
        dim,
        rank);

    if (dim < rank) {
        auto the_shape = t.get_shape();
        auto the_shape_with_padding = t.get_shape().with_tile_padding();
        return the_shape[dim] != the_shape_with_padding[dim];
    }
    return false;
}

inline ttnn::Tensor permute(const ttnn::Tensor& input_tensor, const std::vector<int>& order) {
    const bool initial_input_tensor_on_device = is_on_device(input_tensor);
    const auto input_layout = input_tensor.get_layout();
    const auto input_rank = input_tensor.get_shape().rank();

    TT_FATAL(input_rank <= 4);
    TT_FATAL(
        input_rank == order.size(),
        "The number of dimensions in the tensor input does not match the length of the desired ordering");

    auto adjust_order = [](const std::vector<int>& order) {
        std::vector<std::int64_t> new_order;
        TT_FATAL(order.size() <= 4);
        int additional_ranks = 4 - order.size();
        for (int i = 0; i < additional_ranks; i++) {
            new_order.push_back(i);
        }
        for (int i = 0; i < order.size(); i++) {
            new_order.push_back(order.at(i) + additional_ranks);
        }
        return new_order;
    };
    auto itensor = (input_tensor.get_shape().rank() < 4) ? ttnn::unsqueeze_to_4D(input_tensor) : input_tensor;
    auto iorder = adjust_order(order);

    if (has_tile_padding(itensor)) {
        itensor = ttnn::to_layout(itensor, ttnn::ROW_MAJOR_LAYOUT, std::nullopt, std::nullopt, (Device*)nullptr);
    }

    TT_FATAL(is_on_device(itensor) and itensor.get_shape().rank() == 4);
    auto output_tensor = tt::tt_metal::permute(itensor, iorder, ttnn::DRAM_MEMORY_CONFIG);
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
        output_tensor = ttnn::to_device(output_tensor, input_tensor.device(), ttnn::DRAM_MEMORY_CONFIG);
    }

    return output_tensor;
}

struct UpSample {
    static inline const std::array<TensorSchema, 1> input_tensor_schemas() {
        return {ttnn::TensorSchema{
            2,
            4,
            {ttnn::bfloat16, ttnn::bfloat8_b},
            {ttnn::TILE_LAYOUT, ttnn::ROW_MAJOR_LAYOUT},
            true,
            false,
            false,
            false}};
    }

    template <typename... Args>
    static auto input_tensors_to_validate(const ttnn::Tensor& input_tensor, Args&&... args) {
        return std::make_tuple(input_tensor);
    }

    static ttnn::Tensor execute_on_worker_thread(
        const ttnn::Tensor& input_tensor,
        std::variant<int, std::array<int, 2>, std::array<int, 3>, std::array<int, 4>> scale_factor,
        std::optional<MemoryConfig> output_mem_config = std::nullopt) {
        MemoryConfig mem_config = output_mem_config.value_or(ttnn::DRAM_MEMORY_CONFIG);

        int scale_h = 1;
        int scale_w = 1;
        std::visit(
            [&scale_h, &scale_w](auto&& sf) {
                using T = std::decay_t<decltype(sf)>;
                if constexpr (std::is_same_v<T, int>) {
                    scale_h = sf;
                    scale_w = sf;
                } else if constexpr (std::is_same_v<T, std::array<int, 2>>) {
                    scale_w = sf.at(0);
                    int scale_c = sf.at(1);
                    TT_FATAL(scale_c == 1);
                } else if constexpr (std::is_same_v<T, std::array<int, 3>>) {
                    scale_h = sf.at(0);
                    scale_w = sf.at(1);
                    int scale_c = sf.at(2);
                    TT_FATAL(scale_c == 1);
                } else if constexpr (std::is_same_v<T, std::array<int, 4>>) {
                    int scale_n = sf.at(0);
                    scale_h = sf.at(1);
                    scale_w = sf.at(2);
                    int scale_c = sf.at(3);
                    TT_FATAL(scale_n == 1);
                    TT_FATAL(scale_c == 1);
                } else {
                    // static_assert(false, "Unsupported scale factor");
                    static_assert(sizeof(T) != 0, "Type check failed.");
                }
            },
            scale_factor);

        // DEBUG
        // fmt::print("scale_h: {}, scale_w: {}\n", scale_h, scale_w);

        if (input_tensor.is_sharded()) {
            // TT_FATAL(not input_tensor.is_sharded());
            int shard_height = input_tensor.memory_config().shard_spec.value().shape[0];
            const auto batch_size = input_tensor.get_shape()[0];
            const auto input_h = input_tensor.get_shape()[1];
            const auto input_w = input_tensor.get_shape()[2];
            const auto num_channels = input_tensor.get_shape()[3];
            if (shard_height % input_w != 0) {
                TT_FATAL(shard_height % input_w != 0);
            }
        }

        return tt::tt_metal::upsample(input_tensor, scale_h, scale_w, mem_config);
    }
};

struct Repeat {
    static inline const std::array<TensorSchema, 1> input_tensor_schemas() {
        return {ttnn::TensorSchema{
            4,  // min rank
            4,  // max rank
            {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::int32, ttnn::uint32},
            {ttnn::TILE_LAYOUT, ttnn::ROW_MAJOR_LAYOUT},
            true,     // can_be_on_device
            false,    // can_be_on_cpu
            false,    // can_be_scalar
            false}};  // is_optional
    }

    template <typename... Args>
    static auto input_tensors_to_validate(const ttnn::Tensor& input_tensor, Args&&... args) {
        return std::make_tuple(input_tensor);
    }

    static ttnn::Tensor execute_on_worker_thread(
        const ttnn::Tensor& input_tensor,
        const ttnn::Shape& shape,
        std::optional<MemoryConfig> output_mem_config = std::nullopt) {
        MemoryConfig mem_config = output_mem_config.value_or(input_tensor.memory_config());
        auto output_tensor = tt::tt_metal::repeat(input_tensor, shape.value(), mem_config);
        return output_tensor;
    }
};

struct RepeatInterleave {
    static inline const std::array<TensorSchema, 1> input_tensor_schemas() {
        return {ttnn::TensorSchema{
            4,  // min rank
            4,  // max rank
            {ttnn::bfloat16},
            {ttnn::TILE_LAYOUT},
            true,     // can_be_on_device
            true,    // can_be_on_cpu
            false,    // can_be_scalar
            false}};  // is_optional
    }

    template <typename... Args>
    static auto input_tensors_to_validate(const ttnn::Tensor& input_tensor, Args&&... args) {
        return std::make_tuple(input_tensor);
    }

    // # This operation does not support the following cases:
    // #   - Shape([2[32], 2[32]]) -> repeats = 2, dim = 0
    // #   - Shape([2[32], 2[32]]) -> repeats = Tensor[1,2], dim = 1
    static ttnn::Tensor execute_on_worker_thread(const ttnn::Tensor& input_tensor,
                                                 uint32_t repeats,
                                                 int32_t dim,
                                                 std::optional<MemoryConfig> output_mem_config = std::nullopt) {
        MemoryConfig mem_config = output_mem_config.value_or(input_tensor.memory_config());
        auto output_tensor = tt::tt_metal::repeat_interleave(input_tensor, repeats, dim, mem_config);
        return output_tensor;
    }
};



}  // namespace data_movement
}  // namespace operations

constexpr auto upsample = ttnn::register_operation<ttnn::operations::data_movement::UpSample>("ttnn::upsample");
constexpr auto repeat = ttnn::register_operation<ttnn::operations::data_movement::Repeat>("ttnn::repeat");
constexpr auto repeat_interleave = ttnn::register_operation<ttnn::operations::data_movement::RepeatInterleave>("ttnn::repeat_interleave");

}  // namespace ttnn
