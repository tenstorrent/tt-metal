// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/pixel_unshuffle/pixel_unshuffle.hpp"

namespace ttnn::operations::data_movement {

struct PixelUnshuffle {
    struct operation_attributes_t {
        uint32_t downscale_factor{};
        MemoryConfig output_mem_config{};
        ttnn::PixelUnshuffleChannelOrder channel_order{ttnn::PixelUnshuffleChannelOrder::CHANNEL_MAJOR};
        // channels_last: NHWC output [N, H/r, W/r, padded_channels], HEIGHT_SHARDED L1, written
        // core-locally (MultiCoreChannelsLast). padded_channels >= C*r^2 and a multiple of the L1
        // alignment in elements; the extra channels are zero.
        bool channels_last{false};
        uint32_t padded_channels{0};
    };

    struct tensor_args_t {
        const Tensor& input_tensor;
    };

    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct MultiCore {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output_tensor);
    };

    // NHWC, height-sharded, core-local output; see pixel_unshuffle_channels_last_program_factory.cpp.
    struct MultiCoreChannelsLast {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output_tensor);
    };

    using program_factory_t = std::variant<MultiCore, MultiCoreChannelsLast>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {
ttnn::operations::data_movement::PixelUnshuffle::tensor_return_value_t pixel_unshuffle(
    const ttnn::Tensor& input_tensor,
    uint32_t downscale_factor,
    const MemoryConfig& output_mem_config,
    ttnn::PixelUnshuffleChannelOrder channel_order = ttnn::PixelUnshuffleChannelOrder::CHANNEL_MAJOR,
    bool channels_last = false,
    uint32_t padded_channels = 0);
}  // namespace ttnn::prim
