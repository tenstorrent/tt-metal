// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/transpose/transpose_op.hpp"

#include "ttnn/types.hpp"


namespace ttnn {
namespace operations::data_movement {

namespace permute {
inline bool is_on_device(const Tensor& t) {
    return ttnn::has_storage_type_of(t, ttnn::StorageType::DEVICE) or
    ttnn::has_storage_type_of(t, ttnn::StorageType::MULTI_DEVICE);
}

inline bool has_tile_padding(const Tensor& t) {
    if (t.get_shape().rank() > 1) {
        auto the_shape = t.get_shape();
        auto the_shape_with_padding = t.get_shape().with_tile_padding();
        return the_shape[-1] != the_shape_with_padding[-1] or the_shape[-2] != the_shape_with_padding[-2];
    }
    return false;
}


Tensor permute_launch(const Tensor &a, std::vector<std::int64_t> dims, const MemoryConfig& output_mem_config);
}

struct ExecutePermute {
    static inline ttnn::Tensor composite_invoke(
        uint8_t queue_id,
        const ttnn::Tensor& input_tensor,
        const std::vector<int64_t>& dims,
        const std::optional<MemoryConfig>& memory_config) {
        std::vector<int64_t> iorder(dims.size());

        auto output_tensor =
            permute::permute_launch(input_tensor, dims, memory_config.value_or(input_tensor.memory_config()));

        return output_tensor;
    }

    static inline ttnn::Tensor operator()(
        uint8_t queue_id,
        const ttnn::Tensor& input_tensor,
        const std::vector<int64_t>& dims,
        const std::optional<MemoryConfig>& memory_config,
        bool composite = true) {
        if (composite)
            return composite_invoke(queue_id, input_tensor, dims, memory_config);

        const bool initial_input_tensor_on_device = permute::is_on_device(input_tensor);
        const auto input_layout = input_tensor.get_layout();
        const auto input_rank = input_tensor.get_shape().rank();

        TT_FATAL(input_rank <= 4);
        TT_FATAL(
            input_rank == dims.size(),
            "The number of dimensions in the tensor input does not match the length of the desired ordering");

        auto adjust_order = [](const std::vector<int64_t>& dims) {
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
        auto iorder = adjust_order(dims);  // internals of permute_impl already adjust negative indices

        if (permute::has_tile_padding(itensor)) {
            itensor = ttnn::to_layout(itensor, ttnn::ROW_MAJOR_LAYOUT, std::nullopt, std::nullopt, (Device*)nullptr);
        }

        TT_FATAL(permute::is_on_device(itensor) and itensor.get_shape().rank() == 4);
        auto output_tensor =
            permute::permute_launch(itensor, iorder, memory_config.value_or(input_tensor.memory_config()));
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
            output_tensor = ttnn::reshape(output_tensor, ttnn::Shape::from_vector(shape_vec, full_shape_vec));
        }

        if (initial_input_tensor_on_device and not permute::is_on_device(output_tensor)) {
            output_tensor = ttnn::to_device(
                output_tensor, input_tensor.device(), memory_config.value_or(input_tensor.memory_config()));
        }

        return output_tensor;
    }

    static inline auto operator()(
        const ttnn::Tensor& input_tensor,
        const std::vector<int64_t>& dims,
        const std::optional<MemoryConfig>& memory_config) {
        return operator()(0, input_tensor, dims, memory_config);
    }

    static inline auto operator()(const ttnn::Tensor& input_tensor, const std::vector<int64_t>& dims) {
        return operator()(input_tensor, dims, std::nullopt);
    }
};

}  // namespace operations::data_movement

constexpr auto permute = ttnn::register_operation<"ttnn::permute", ttnn::operations::data_movement::ExecutePermute>();

}  // namespace ttnn
