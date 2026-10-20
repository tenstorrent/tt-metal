// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "kda_decode_device_operation.hpp"

#include <initializer_list>

#include <tt-metalium/constants.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

KdaDecodeDeviceOperation::program_factory_t KdaDecodeDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return KdaDecodeProgramFactory{};
}

void KdaDecodeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    using namespace tt::constants;
    const uint32_t B = attrs.B;
    const uint32_t H = attrs.H;
    const uint32_t K = attrs.key_dim;
    const uint32_t V = attrs.val_dim;
    TT_FATAL(B >= 1 && H >= 1, "kda_decode: B ({}) and H ({}) must be >= 1", B, H);
    TT_FATAL(
        K >= TILE_WIDTH && K % TILE_WIDTH == 0 && V >= TILE_WIDTH && V % TILE_WIDTH == 0,
        "kda_decode: K ({}) and V ({}) must be non-zero multiples of {}",
        K,
        V,
        TILE_WIDTH);
    const uint32_t Hb = ((H + TILE_WIDTH - 1) / TILE_WIDTH) * TILE_WIDTH;
    auto* device = in.qkv.device();
    const bool fp32_dest_acc_en =
        std::get<2>(get_compute_kernel_config_args(device->arch(), attrs.compute_kernel_config));
    TT_FATAL(fp32_dest_acc_en, "kda_decode: requires fp32_dest_acc_en (the state is carried in fp32 DST)");

    auto check = [&](const Tensor& t,
                     const char* name,
                     DataType dtype,
                     Layout layout,
                     std::initializer_list<uint32_t> expected) {
        TT_FATAL(
            t.storage_type() == StorageType::DEVICE && t.buffer() != nullptr,
            "kda_decode: {} must be allocated on device",
            name);
        TT_FATAL(t.device() == device, "kda_decode: {} must be on the same device as qkv", name);
        TT_FATAL(!t.is_sharded(), "kda_decode: {} must be interleaved, not sharded", name);
        TT_FATAL(t.layout() == layout, "kda_decode: {} must be {} layout", name, layout);
        TT_FATAL(t.dtype() == dtype, "kda_decode: {} must be {}, got {}", name, dtype, t.dtype());
        const auto& shape = t.logical_shape();
        TT_FATAL(
            shape.rank() == expected.size(),
            "kda_decode: {} must be rank {}, got rank {}",
            name,
            expected.size(),
            shape.rank());
        size_t i = 0;
        for (uint32_t e : expected) {
            TT_FATAL(
                static_cast<uint32_t>(shape[i]) == e,
                "kda_decode: {} dim[{}] must be {}, got {}",
                name,
                i,
                e,
                shape[i]);
            ++i;
        }
    };
    check(in.qkv, "qkv", DataType::BFLOAT16, Layout::TILE, {1u, 1u, B, 3u * H * K});
    check(in.gates, "gates", DataType::FLOAT32, Layout::TILE, {1u, 1u, B, H * K + H * V + Hb});
    check(in.conv_state, "conv_state", DataType::BFLOAT16, Layout::ROW_MAJOR, {B, 3u, 3u * H * K});
    check(in.recurrent, "recurrent", DataType::FLOAT32, Layout::TILE, {B, H, K, V});
    check(in.taps, "taps", DataType::FLOAT32, Layout::TILE, {1u, 1u, 4u, 3u * H * K});
    check(in.chan_consts, "chan_consts", DataType::FLOAT32, Layout::TILE, {1u, 1u, 2u, H * K});
    check(in.o_norm_w, "o_norm_w", DataType::FLOAT32, Layout::TILE, {1u, 1u, 1u, V});
    check(in.selectors, "selectors", DataType::BFLOAT16, Layout::TILE, {33u * TILE_HEIGHT, TILE_WIDTH});
}

KdaDecodeDeviceOperation::spec_return_value_t KdaDecodeDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    const auto layout = TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), attrs.output_mem_config);
    ttnn::Shape o_shape({1, 1, attrs.B, attrs.H * attrs.val_dim});
    return {tt::tt_metal::TensorSpec(o_shape, layout), in.recurrent.tensor_spec(), in.conv_state.tensor_spec()};
}

KdaDecodeDeviceOperation::tensor_return_value_t KdaDecodeDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    auto specs = compute_output_specs(attrs, in);
    // the state outputs alias the caller's tensors (updated in place)
    return {create_device_tensor(specs[0], in.qkv.device()), in.recurrent, in.conv_state};
}

std::vector<Tensor> kda_decode(
    const Tensor& qkv,
    const Tensor& gates,
    const Tensor& conv_state,
    const Tensor& recurrent,
    const Tensor& taps,
    const Tensor& chan_consts,
    const Tensor& o_norm_w,
    const Tensor& selectors,
    uint32_t num_heads,
    float gate_lower_bound,
    float l2_eps,
    float norm_eps,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    using OperationType = KdaDecodeDeviceOperation;
    const auto& rs = recurrent.logical_shape();  // [B, H, K, V]
    TT_FATAL(rs.rank() == 4, "kda_decode: recurrent must be rank 4 [B, H, K, V], got rank {}", rs.rank());
    TT_FATAL(rs[1] == num_heads, "kda_decode: recurrent dim 1 ({}) must equal num_heads ({})", rs[1], num_heads);
    auto attrs = OperationType::operation_attributes_t{
        .B = static_cast<uint32_t>(rs[0]),
        .H = num_heads,
        .key_dim = static_cast<uint32_t>(rs[2]),
        .val_dim = static_cast<uint32_t>(rs[3]),
        .gate_lower_bound = gate_lower_bound,
        .l2_eps = l2_eps,
        .norm_eps = norm_eps,
        .output_mem_config = output_mem_config,
        .compute_kernel_config = compute_kernel_config,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .qkv = qkv,
        .gates = gates,
        .conv_state = conv_state,
        .recurrent = recurrent,
        .taps = taps,
        .chan_consts = chan_consts,
        .o_norm_w = o_norm_w,
        .selectors = selectors,
    };
    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

}  // namespace ttnn::prim
