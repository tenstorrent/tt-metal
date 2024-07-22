// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/types.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/concat/concat_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/repeat/repeat_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/composite/composite_ops.hpp"
#include "ttnn/operations/upsample/upsample_op.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"

#include <ranges>

namespace ttnn {
namespace operations {
namespace data_movement {

// repeat interleave supports repeats as 1 to inf, dim between 0 to 2
inline Tensor repeat_interleave(const Tensor& input_a, uint32_t repeat, int32_t dim, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> combined_tensors;
    combined_tensors.reserve(repeat);
    auto shape_wh = input_a.get_legacy_shape();
    // normalizing the negative dim
    uint32_t normalized_dim = input_a.get_legacy_shape().get_normalized_index(dim);
    // check if dim is 1 or 3
    if (normalized_dim & 1) {
        constexpr uint32_t tmp_dim = 2;
        std::vector<int64_t> dims = {0, 1, 2, 3};
        std::swap(dims[dim], dims[tmp_dim]);
        Tensor transpose_input = ttnn::permute(input_a, dims);
        Tensor ril_result = repeat_interleave(transpose_input, repeat, tmp_dim, output_mem_config);
        return ttnn::permute(ril_result, dims);
    }
    if (normalized_dim <= 1) {
        for (int i = 0; i < repeat; i++) {
            combined_tensors.push_back(input_a);
        }
        // TODO: For dim = 1 facing issue with concat_op
        if (normalized_dim) {
            Tensor concat_out = concat(combined_tensors, 2);
            return reshape(concat_out, shape_wh[0], shape_wh[1] * repeat, shape_wh[2], shape_wh[3]);
        } else {
            Tensor concat_out = concat(combined_tensors, 1);
            return reshape(concat_out, shape_wh[0] * repeat, shape_wh[1], shape_wh[2], shape_wh[3]);
        }
    } else {
        Tensor reshape_out =
            reshape(input_a, 1, 1, shape_wh[0] * shape_wh[1] * shape_wh[2], shape_wh[3]);
        for (int i = 0; i < repeat; i++) {
            combined_tensors.push_back(reshape_out);
        }
        Tensor concat_out = concat(combined_tensors, 1);
        std::vector<int64_t> permute_dims = {0, 2, 1, 3};
        Tensor permute_out = ttnn::permute(concat_out, permute_dims);
        return reshape(permute_out, shape_wh[0], shape_wh[1], shape_wh[2] * repeat, shape_wh[3]);
    }
}

struct UpSample {

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

    // # This operation does not support the following cases:
    // #   - Shape([2[32], 2[32]]) -> repeats = 2, dim = 0
    // #   - Shape([2[32], 2[32]]) -> repeats = Tensor[1,2], dim = 1
    static ttnn::Tensor execute_on_worker_thread(const ttnn::Tensor& input_tensor,
                                                 uint32_t repeats,
                                                 int32_t dim,
                                                 std::optional<MemoryConfig> output_mem_config = std::nullopt) {
        MemoryConfig mem_config = output_mem_config.value_or(input_tensor.memory_config());
        auto output_tensor = repeat_interleave(input_tensor, repeats, dim, mem_config);
        return output_tensor;
    }
};

}  // namespace data_movement
}  // namespace operations

constexpr auto upsample = ttnn::register_operation<ttnn::operations::data_movement::UpSample>("ttnn::upsample");
constexpr auto repeat = ttnn::register_operation<ttnn::operations::data_movement::Repeat>("ttnn::repeat");
constexpr auto repeat_interleave = ttnn::register_operation<ttnn::operations::data_movement::RepeatInterleave>("ttnn::repeat_interleave");

}  // namespace ttnn
