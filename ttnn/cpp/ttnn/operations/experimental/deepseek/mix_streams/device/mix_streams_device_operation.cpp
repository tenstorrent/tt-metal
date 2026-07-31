// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mix_streams_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::deepseek::mix_streams {

namespace {

// The fused kernel keeps comb (and its transpose) in a single tile and folds the
// placement outer product into a second matmul against the same DST accumulator,
// which is what pins hc to one tile and D to a whole number of tiles.
struct Dims {
    uint32_t b = 0;
    uint32_t s = 0;
    uint32_t hc = 0;
    uint32_t d = 0;
};

std::optional<Dims> fusable_dims(
    const Tensor& post, const Tensor& comb, const Tensor& sublayer_out, const Tensor& streams) {
    for (const Tensor* t : {&post, &comb, &sublayer_out, &streams}) {
        if (t->storage_type() != StorageType::DEVICE || t->layout() != Layout::TILE) {
            return std::nullopt;
        }
        // The reader assembles the comb / post tiles element-wise in L1, which is
        // written for a 2-byte element type.
        if (t->dtype() != DataType::BFLOAT16) {
            return std::nullopt;
        }
        if (t->logical_shape().rank() != 4) {
            return std::nullopt;
        }
    }

    const auto& streams_shape = streams.logical_shape();
    const Dims dims{
        static_cast<uint32_t>(streams_shape[0]),
        static_cast<uint32_t>(streams_shape[1]),
        static_cast<uint32_t>(streams_shape[2]),
        static_cast<uint32_t>(streams_shape[3])};

    if (dims.hc == 0 || dims.hc > tt::constants::TILE_HEIGHT || dims.d % tt::constants::TILE_WIDTH != 0) {
        return std::nullopt;
    }
    const bool shapes_ok = post.logical_shape() == ttnn::Shape({dims.b, dims.s, dims.hc, 1}) &&
                           comb.logical_shape() == ttnn::Shape({dims.b, dims.s, dims.hc, dims.hc}) &&
                           sublayer_out.logical_shape() == ttnn::Shape({dims.b, dims.s, 1, dims.d});
    if (!shapes_ok) {
        return std::nullopt;
    }
    // The kernel walks tensors page-by-page assuming the standard 32x32 tile.
    for (const Tensor* t : {&post, &comb, &sublayer_out, &streams}) {
        const auto& tile = t->tensor_spec().tile();
        if (tile.get_height() != tt::constants::TILE_HEIGHT || tile.get_width() != tt::constants::TILE_WIDTH) {
            return std::nullopt;
        }
    }
    return dims;
}

void validate_tensors(const MixStreamsParams& attributes, const MixStreamsInputs& tensor_args) {
    const auto dims = fusable_dims(tensor_args.post, tensor_args.comb, tensor_args.sublayer_out, tensor_args.streams);
    TT_FATAL(
        dims.has_value(),
        "mix_streams: inputs are not supported by the fused kernel -- expected device-resident, TILE-layout, "
        "BFLOAT16 tensors shaped post [B,S,hc,1], comb [B,S,hc,hc], sublayer_out [B,S,1,D], streams [B,S,hc,D] "
        "with hc <= {} and D a multiple of {}; got post {}, comb {}, sublayer_out {}, streams {}",
        tt::constants::TILE_HEIGHT,
        tt::constants::TILE_WIDTH,
        tensor_args.post.logical_shape(),
        tensor_args.comb.logical_shape(),
        tensor_args.sublayer_out.logical_shape(),
        tensor_args.streams.logical_shape());
    TT_FATAL(
        attributes.num_streams == dims->hc,
        "mix_streams: num_streams {} must match the streams stream dim {}",
        attributes.num_streams,
        dims->hc);
}

}  // namespace

bool is_fusable(const Tensor& post, const Tensor& comb, const Tensor& sublayer_out, const Tensor& streams) {
    return fusable_dims(post, comb, sublayer_out, streams).has_value();
}

void MixStreamsDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_tensors(attributes, tensor_args);
}

void MixStreamsDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_tensors(attributes, tensor_args);
}

MixStreamsDeviceOperation::spec_return_value_t MixStreamsDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& streams = tensor_args.streams;
    return tt::tt_metal::TensorSpec(
        streams.logical_shape(),
        tt::tt_metal::TensorLayout(
            streams.dtype(),
            tt::tt_metal::PageConfig(Layout::TILE, streams.tensor_spec().tile()),
            operation_attributes.output_mem_config));
}

MixStreamsDeviceOperation::tensor_return_value_t MixStreamsDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.streams.device());
}

}  // namespace ttnn::operations::experimental::deepseek::mix_streams

namespace ttnn::prim {

Tensor mix_streams(
    const Tensor& post,
    const Tensor& comb,
    const Tensor& sublayer_out,
    const Tensor& streams,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config) {
    using OperationType = ttnn::operations::experimental::deepseek::mix_streams::MixStreamsDeviceOperation;

    // HiFi4 / fp32 dest acc / packer-l1-acc by default, matching the ``_HIFI4`` config
    // the eager Python path passes to ttnn.matmul.
    const auto kernel_config = init_device_compute_kernel_config(
        streams.device()->arch(),
        compute_kernel_config,
        /*default_fidelity=*/MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/true);

    auto operation_attributes = OperationType::operation_attributes_t{
        .num_streams = static_cast<uint32_t>(streams.logical_shape()[2]),
        .output_mem_config = memory_config.value_or(streams.memory_config()),
        .compute_kernel_config = kernel_config,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .post = post,
        .comb = comb,
        .sublayer_out = sublayer_out,
        .streams = streams,
    };
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
