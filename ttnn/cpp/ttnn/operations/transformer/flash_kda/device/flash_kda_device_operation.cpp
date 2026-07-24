// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/flash_kda/device/flash_kda_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operation.hpp"
#include <tt-metalium/constants.hpp>
#include <cstdint>

using namespace tt::tt_metal;

namespace ttnn::prim {

static void validate_tensor(const Tensor& t, const std::string& name) {
    TT_FATAL(t.storage_type() == StorageType::DEVICE, "{} must be on device", name);
    TT_FATAL(t.buffer() != nullptr, "{} must be allocated", name);
    TT_FATAL(t.buffer()->buffer_type() == BufferType::DRAM, "{} must be in DRAM", name);
    TT_FATAL(t.layout() == Layout::TILE, "{} must be tiled", name);
    TT_FATAL(t.dtype() == DataType::FLOAT32, "{} must be float32, got {}", name, t.dtype());
}

void FlashKdaDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    validate_tensor(in.S_prev, "S_prev");
    validate_tensor(in.g, "g");
    validate_tensor(in.k, "k");
    validate_tensor(in.v, "v");
    validate_tensor(in.beta, "beta");
    validate_tensor(in.q, "q");

    const std::uint32_t N = attrs.num_items;
    const std::uint32_t Dk = in.k.logical_shape()[2];
    const std::uint32_t Dv = in.v.logical_shape()[2];

    // The compute kernel's loops are generic over Kt = Dk/32, Vt = Dv/32 tiles (unlike
    // gated_delta_attn's kernel, which hardwires exactly 4 block-rows), so only tile
    // alignment is required here, not a fixed magic-number dimension.
    TT_FATAL(Dk % tt::constants::TILE_WIDTH == 0, "key_dim (Dk={}) must be a multiple of 32", Dk);
    TT_FATAL(Dv % tt::constants::TILE_WIDTH == 0, "val_dim (Dv={}) must be a multiple of 32", Dv);

    auto check_shape = [&](const Tensor& t, std::initializer_list<std::uint32_t> expected, const std::string& nm) {
        auto s = t.logical_shape();
        TT_FATAL(s.rank() == expected.size(), "{} rank mismatch: {} vs {}", nm, s.rank(), expected.size());
        size_t i = 0;
        for (auto e : expected) {
            TT_FATAL(static_cast<std::uint32_t>(s[i]) == e, "{} dim[{}] expected {} got {}", nm, i, e, s[i]);
            i++;
        }
    };

    check_shape(in.S_prev, {N, Dk, Dv}, "S_prev");
    check_shape(in.g, {N, Dk, 1}, "g");
    check_shape(in.k, {N, 1, Dk}, "k");
    check_shape(in.v, {N, 1, Dv}, "v");
    check_shape(in.beta, {N, 1, 1}, "beta");
    check_shape(in.q, {N, 1, Dk}, "q");
}

void FlashKdaDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    validate_on_program_cache_miss(attrs, in);
}

FlashKdaDeviceOperation::spec_return_value_t FlashKdaDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    const std::uint32_t N = attrs.num_items;
    const std::uint32_t Dk = in.k.logical_shape()[2];
    const std::uint32_t Dv = in.v.logical_shape()[2];
    const auto& mc = attrs.output_mem_config;

    tt::tt_metal::TensorSpec s_new_spec(
        ttnn::Shape({N, Dk, Dv}), TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), mc));
    tt::tt_metal::TensorSpec out_spec(
        ttnn::Shape({N, 1, Dv}), TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), mc));
    return {s_new_spec, out_spec};
}

FlashKdaDeviceOperation::tensor_return_value_t FlashKdaDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    auto specs = compute_output_specs(attrs, in);
    return {
        create_device_tensor(specs[0], in.S_prev.device()),
        create_device_tensor(specs[1], in.S_prev.device()),
    };
}

ttsl::hash::hash_t FlashKdaDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    return operation::hash_operation<FlashKdaDeviceOperation>(
        attrs.num_items,
        attrs.output_mem_config,
        attrs.compute_kernel_config,
        in.S_prev,
        in.g,
        in.k,
        in.v,
        in.beta,
        in.q);
}

std::vector<Tensor> flash_kda(
    const Tensor& S_prev,
    const Tensor& g,
    const Tensor& k,
    const Tensor& v,
    const Tensor& beta,
    const Tensor& q,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    using Op = FlashKdaDeviceOperation;
    return ttnn::device_operation::launch<Op>(
        Op::operation_attributes_t{
            .num_items = static_cast<std::uint32_t>(S_prev.logical_shape()[0]),
            .output_mem_config = output_mem_config,
            .compute_kernel_config = compute_kernel_config,
        },
        Op::tensor_args_t{
            .S_prev = S_prev,
            .g = g,
            .k = k,
            .v = v,
            .beta = beta,
            .q = q,
        });
}

}  // namespace ttnn::prim
