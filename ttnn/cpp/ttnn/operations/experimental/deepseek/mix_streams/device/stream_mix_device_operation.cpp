// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "stream_mix_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::deepseek::mix_streams {

namespace {

// The reader takes each token's weights out of the first 16x16 face of its tile, so hc fits a face.
constexpr uint32_t kStreamMixMaxStreams = 16;

void check_operand(const Tensor& t, const char* name) {
    TT_FATAL(t.storage_type() == StorageType::DEVICE, "stream_mix: {} must be on device", name);
    TT_FATAL(t.dtype() == DataType::BFLOAT16, "stream_mix: {} must be BFLOAT16, got {}", name, t.dtype());
    TT_FATAL(t.layout() == Layout::TILE, "stream_mix: {} must be TILE layout", name);
    const auto& tile = t.tensor_spec().tile();
    TT_FATAL(
        tile.get_height() == tt::constants::TILE_HEIGHT && tile.get_width() == tt::constants::TILE_WIDTH,
        "stream_mix: {} must use 32x32 tiles",
        name);
    TT_FATAL(
        t.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "stream_mix: {} must be interleaved",
        name);
    TT_FATAL(t.logical_shape().rank() == 4, "stream_mix: {} must be rank 4, got {}", name, t.logical_shape());
}

void validate_stream_mix(const StreamMixParams& attributes, const StreamMixInputs& args) {
    check_operand(args.streams, "streams");
    check_operand(args.weights, "weights");
    TT_FATAL(
        args.post.has_value() == args.sublayer_out.has_value(),
        "stream_mix: post and sublayer_out are given together or not at all");

    const auto& s = args.streams.logical_shape();
    const uint32_t hc = s[1];
    const uint32_t t = s[2];
    const uint32_t d = s[3];
    TT_FATAL(s[0] == 1, "stream_mix: streams must be [1, hc, T, D], got {}", s);
    TT_FATAL(
        hc >= 1 && hc <= kStreamMixMaxStreams, "stream_mix: hc must be in [1, {}], got {}", kStreamMixMaxStreams, hc);
    TT_FATAL(d % tt::constants::TILE_WIDTH == 0, "stream_mix: D ({}) must be a multiple of 32", d);
    TT_FATAL(
        attributes.num_streams == hc, "stream_mix: num_streams {} != streams dim 1 {}", attributes.num_streams, hc);

    const uint32_t n_out = attributes.num_outputs;
    const ttnn::Shape weights_shape({1, t, n_out == 1 ? 1u : hc, hc});
    TT_FATAL(
        args.weights.logical_shape() == weights_shape,
        "stream_mix: weights must be pre [1, T, 1, hc] or comb [1, T, hc, hc]; expected {}, got {}",
        weights_shape,
        args.weights.logical_shape());

    if (args.post.has_value()) {
        check_operand(*args.post, "post");
        check_operand(*args.sublayer_out, "sublayer_out");
        TT_FATAL(
            args.post->logical_shape() == ttnn::Shape({1, t, n_out, 1}),
            "stream_mix: post must be [1, T, n_out, 1] = [1, {}, {}, 1], got {}",
            t,
            n_out,
            args.post->logical_shape());
        TT_FATAL(
            args.sublayer_out->logical_shape() == ttnn::Shape({1, 1, t, d}),
            "stream_mix: sublayer_out must be [1, 1, T, D] = [1, 1, {}, {}], got {}",
            t,
            d,
            args.sublayer_out->logical_shape());
    }
}

}  // namespace

void StreamMixDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_stream_mix(attributes, tensor_args);
}

void StreamMixDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_stream_mix(attributes, tensor_args);
}

StreamMixDeviceOperation::spec_return_value_t StreamMixDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& streams = tensor_args.streams;
    const auto& s = streams.logical_shape();
    return tt::tt_metal::TensorSpec(
        ttnn::Shape({1, attributes.num_outputs, s[2], s[3]}),
        tt::tt_metal::TensorLayout(
            streams.dtype(),
            tt::tt_metal::PageConfig(Layout::TILE, streams.tensor_spec().tile()),
            attributes.output_mem_config));
}

StreamMixDeviceOperation::tensor_return_value_t StreamMixDeviceOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(attributes, tensor_args), tensor_args.streams.device());
}

}  // namespace ttnn::operations::experimental::deepseek::mix_streams

namespace ttnn::prim {

Tensor stream_mix(
    const Tensor& streams,
    const Tensor& weights,
    const std::optional<Tensor>& post,
    const std::optional<Tensor>& sublayer_out,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config) {
    using OperationType = ttnn::operations::experimental::deepseek::mix_streams::StreamMixDeviceOperation;

    const auto kernel_config = init_device_compute_kernel_config(
        streams.device()->arch(),
        compute_kernel_config,
        /*default_fidelity=*/tt::tt_metal::MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false);

    const uint32_t hc = static_cast<uint32_t>(streams.logical_shape()[1]);
    auto operation_attributes = OperationType::operation_attributes_t{
        .num_streams = hc,
        .num_outputs = weights.logical_shape()[2] == 1 ? 1u : hc,
        .output_mem_config = memory_config.value_or(streams.memory_config()),
        .compute_kernel_config = kernel_config,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .streams = streams,
        .weights = weights,
        .post = post,
        .sublayer_out = sublayer_out,
    };
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
