// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "upsample.hpp"
#include "device/upsample_op.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::upsample {

ttnn::Tensor ExecuteUpSample::invoke(const ttnn::Tensor& input_tensor,
    std::variant<int, tt::tt_metal::Array2D, tt::tt_metal::Array3D, tt::tt_metal::Array4D> scale_factor,
    const std::optional<MemoryConfig>& output_mem_config) {
        MemoryConfig mem_config = output_mem_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
        int scale_h = 1;
        int scale_w = 1;
        std::visit(
            [&scale_h, &scale_w](auto&& sf) {
                using T = std::decay_t<decltype(sf)>;
                if constexpr (std::is_same_v<T, int>) {
                    scale_h = sf;
                    scale_w = sf;
                } else if constexpr (std::is_same_v<T, tt::tt_metal::Array2D>) {
                    scale_w = sf.at(0);
                    int scale_c = sf.at(1);
                    TT_FATAL(scale_c == 1, "Error");
                } else if constexpr (std::is_same_v<T, tt::tt_metal::Array3D>) {
                    scale_h = sf.at(0);
                    scale_w = sf.at(1);
                    int scale_c = sf.at(2);
                    TT_FATAL(scale_c == 1, "Error");
                } else if constexpr (std::is_same_v<T, tt::tt_metal::Array4D>) {
                    int scale_n = sf.at(0);
                    scale_h = sf.at(1);
                    scale_w = sf.at(2);
                    int scale_c = sf.at(3);
                    TT_FATAL(scale_n == 1, "Error");
                    TT_FATAL(scale_c == 1, "Error");
                } else {
                    // static_assert(false, "Unsupported scale factor");
                    static_assert(sizeof(T) != 0, "Type check failed.");
                }
            },
            scale_factor);

        // DEBUG
        // fmt::print("scale_h: {}, scale_w: {}\n", scale_h, scale_w);

        if (input_tensor.is_sharded()) {
            // TT_FATAL(not input_tensor.is_sharded(), "Error");
            int shard_height = input_tensor.memory_config().shard_spec.value().shape[0];
            const auto batch_size = input_tensor.get_shape()[0];
            const auto input_h = input_tensor.get_shape()[1];
            const auto input_w = input_tensor.get_shape()[2];
            const auto num_channels = input_tensor.get_shape()[3];
            if (shard_height % input_w != 0) {
                TT_FATAL(shard_height % input_w != 0, "Error");
            }
        }

        //return ttnn::upsample(input_tensor, scale_h, scale_w, mem_config);
        auto output_tensor = operation::run(
            UpSample{scale_h, scale_w, mem_config},
            {input_tensor}).front();
        return output_tensor;
    }

}  // namespace upsample
