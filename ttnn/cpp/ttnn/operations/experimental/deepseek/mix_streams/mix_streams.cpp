// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mix_streams.hpp"

#include "ttnn/operations/core/to_memory_config/to_memory_config_op.hpp"
#include "ttnn/operations/data_movement/repeat/repeat.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/matmul/matmul.hpp"

namespace ttnn::experimental::deepseek::mix_streams {

namespace {

// HiFi4 / fp32-dest-acc / packer-l1-acc -- mirrors ``_HIFI4`` in
// models/experimental/deepseek_v4_flash/tt/common.py so the fused matmul
// matches the eager ``ttnn.matmul(..., compute_kernel_config=_HIFI4)`` path.
DeviceComputeKernelConfig default_hifi4_config() {
    DeviceComputeKernelConfig cfg{};
    cfg.math_fidelity = tt::tt_metal::MathFidelity::HiFi4;
    cfg.math_approx_mode = false;
    cfg.fp32_dest_acc_en = true;
    cfg.packer_l1_acc = true;
    return cfg;
}

void validate_inputs(const Tensor& post, const Tensor& comb, const Tensor& sublayer_out, const Tensor& streams) {
    TT_FATAL(post.storage_type() == StorageType::DEVICE, "mix_streams: post must be on device");
    TT_FATAL(comb.storage_type() == StorageType::DEVICE, "mix_streams: comb must be on device");
    TT_FATAL(sublayer_out.storage_type() == StorageType::DEVICE, "mix_streams: sublayer_out must be on device");
    TT_FATAL(streams.storage_type() == StorageType::DEVICE, "mix_streams: streams must be on device");

    const auto& streams_shape = streams.logical_shape();
    TT_FATAL(
        streams_shape.rank() == 4,
        "mix_streams: streams must be rank-4 [B, S, hc, D], got rank {}",
        streams_shape.rank());

    const uint32_t b = static_cast<uint32_t>(streams_shape[0]);
    const uint32_t s = static_cast<uint32_t>(streams_shape[1]);
    const uint32_t hc = static_cast<uint32_t>(streams_shape[2]);
    const uint32_t d = static_cast<uint32_t>(streams_shape[3]);

    const auto& post_shape = post.logical_shape();
    const auto& comb_shape = comb.logical_shape();
    const auto& out_shape = sublayer_out.logical_shape();

    TT_FATAL(
        post_shape.rank() == 4 && static_cast<uint32_t>(post_shape[0]) == b &&
            static_cast<uint32_t>(post_shape[1]) == s && static_cast<uint32_t>(post_shape[2]) == hc &&
            static_cast<uint32_t>(post_shape[3]) == 1,
        "mix_streams: post must be [B, S, hc, 1] = [{}, {}, {}, 1], got {}",
        b,
        s,
        hc,
        post_shape);
    TT_FATAL(
        comb_shape.rank() == 4 && static_cast<uint32_t>(comb_shape[0]) == b &&
            static_cast<uint32_t>(comb_shape[1]) == s && static_cast<uint32_t>(comb_shape[2]) == hc &&
            static_cast<uint32_t>(comb_shape[3]) == hc,
        "mix_streams: comb must be [B, S, hc, hc] = [{}, {}, {}, {}], got {}",
        b,
        s,
        hc,
        hc,
        comb_shape);
    TT_FATAL(
        out_shape.rank() == 4 && static_cast<uint32_t>(out_shape[0]) == b && static_cast<uint32_t>(out_shape[1]) == s &&
            static_cast<uint32_t>(out_shape[2]) == 1 && static_cast<uint32_t>(out_shape[3]) == d,
        "mix_streams: sublayer_out must be [B, S, 1, D] = [{}, {}, 1, {}], got {}",
        b,
        s,
        d,
        out_shape);
}

}  // namespace

Tensor mix_streams(
    const Tensor& post,
    const Tensor& comb,
    const Tensor& sublayer_out,
    const Tensor& streams,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    validate_inputs(post, comb, sublayer_out, streams);

    const auto& streams_shape = streams.logical_shape();
    const uint32_t b = static_cast<uint32_t>(streams_shape[0]);
    const uint32_t s = static_cast<uint32_t>(streams_shape[1]);
    const uint32_t hc = static_cast<uint32_t>(streams_shape[2]);
    const uint32_t d = static_cast<uint32_t>(streams_shape[3]);
    const uint32_t t = b * s;

    // placement = post[..,None] * sublayer_out[..,None,:] -> [1, T, hc, D].
    auto out = ttnn::reshape(sublayer_out, ttnn::Shape({1, t, 1, d}));
    out = ttnn::repeat(out, ttnn::Shape({1, 1, hc, 1}));  // broadcast over the stream axis
    auto placement = ttnn::multiply(out, ttnn::reshape(post, ttnn::Shape({1, t, hc, 1})));

    // mixed = matmul(comb^T, streams): sum over the FIRST hc axis. Fold the comb
    // transpose into the matmul (transpose_a=True) to drop a separate transpose
    // device op -- at 4x4 the transpose is ~30us of dispatch overhead for ~2us of
    // compute, so the op is pure launch cost.
    auto comb_r = ttnn::reshape(comb, ttnn::Shape({1, t, hc, hc}));
    auto streams_r = ttnn::reshape(streams, ttnn::Shape({1, t, hc, d}));
    auto mixed = ttnn::matmul(
        /*input_tensor_a=*/comb_r,
        /*input_tensor_b=*/streams_r,
        /*transpose_a=*/true,
        /*transpose_b=*/false,
        /*memory_config=*/std::nullopt,
        /*dtype=*/std::nullopt,
        /*program_config=*/std::nullopt,
        /*activation=*/std::nullopt,
        /*compute_kernel_config=*/compute_kernel_config.value_or(default_hifi4_config()));

    auto result = ttnn::reshape(ttnn::add(placement, mixed), ttnn::Shape({b, s, hc, d}));
    if (memory_config.has_value()) {
        result = ttnn::to_memory_config(result, *memory_config);
    }
    return result;
}

}  // namespace ttnn::experimental::deepseek::mix_streams
